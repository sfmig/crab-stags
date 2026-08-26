"""Detect STag markers live on a FLIR (Spinnaker) camera and log their centres.

The FLIR sibling of `scripts/detect_markers.py`: OpenCV's VideoCapture cannot
open a machine-vision camera, so this goes through EasyPySpin
(https://github.com/elerac/EasyPySpin), a thin wrapper that gives FLIR's PySpin
a cv2.VideoCapture-like interface.

Needs the Spinnaker SDK and its PySpin wheel (both from FLIR, not on PyPI),
then:
    pip install EasyPySpin

Driven by a YAML config file (see `configs/detect_markers_flir_config.yaml`):

    python scripts/detect_markers_flir.py configs/detect_markers_flir_config.yaml

The camera is taken as it is set up: this script does not touch trigger mode
or any other persistent camera state beyond the settings named in the config.
A camera in trigger mode therefore delivers nothing unless something triggers
it, and the run ends on a PySpin grab timeout instead of quietly rewiring it.

Writes one CSV row per detected marker per frame (frame, marker_id, centre_x,
centre_y), and optionally an annotated .mp4 alongside it. Both go to the repo's
output/ folder unless the config overrides `output_dir`, and both are named
after the camera serial, so two cameras watching the same scene stay apart.

This one reads a live camera rather than a file, so it runs until you stop it:
press 'q' in the preview window, or Ctrl-C, or set `max_frames`. Whatever was
detected up to that point is still saved.

The preview window also tunes the camera while it streams -- g/h gain, e/r
exposure, t/y gamma, b/n black level -- so you can watch what a setting does to
the detections instead of guessing, restarting, and guessing again. The values
you land on are printed at the end in config-file form, ready to paste back.

The centre is the intersection of the marker's image diagonals rather than the
mean of its corners; see `crab_stags/markers.py` for why. Detection runs in a
subprocess because stag-python segfaults on some real frames; see
`crab_stags/stag_safe.py`.

To list the connected cameras and their serial numbers:
    python scripts/detect_markers_flir.py configs/detect_markers_flir_config.yaml --list-cameras
"""

import argparse
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import stag
import yaml
from caliscope.api import CameraArray

from crab_stags.markers import marker_centres
from crab_stags.stag_safe import StagDetector

try:
    import EasyPySpin
    import PySpin
except ImportError as err:  # so --help still works without the SDK installed
    EasyPySpin = PySpin = None
    IMPORT_ERROR: ImportError | None = err
else:
    IMPORT_ERROR = None

# Repo-level output folder, a sibling of scripts/
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"


@dataclass
class Config:
    """Inputs for a live FLIR marker detection run."""

    # Serial number of the camera to open, as a string (run with
    # --list-cameras to see them). None takes the first camera found.
    camera_serial: str | None = None

    # Camera intrinsics from scripts/compute_intrinsics.py. Optional but recommended,
    # used to undistort the marker corners before the centre is computed.
    # Must describe a single camera (see load_camera).
    intrinsics: Path | None = None

    # STag library/family: one of [11, 13, 15, 17, 19, 21, 23]
    # see https://github.com/ManfredStoiber/stag-python#-configuration
    stag_libraryHD: int = 15

    # Requested capture settings -- None keeps whatever the camera currently has.
    # EasyPySpin clips out-of-range values, so we use what it reports back.
    frame_width: int | None = None
    frame_height: int | None = None
    fps: float | None = None
    exposure_us: float | None = None
    gain_db: float | None = None
    gamma: float | None = None
    black_level: float | None = None

    # True (the default here) always detects on the newest frame the camera has,
    # dropping whatever piled up while the previous one was being processed.
    # Detection is slower than acquisition, so False makes the run fall further
    # and further behind the present -- only worth it if you need every frame.
    buffer_newest_only: bool = True

    # How long to wait for a frame before giving up
    grab_timeout_ms: int = 5000

    # Outputs
    # Defaults to DEFAULT_OUTPUT_DIR
    output_dir: Path | None = None
    save_annotated_video: bool = True

    # Preview
    # press q to stop
    show_preview: bool = True

    # Stop after this many frames. None runs until 'q' or Ctrl-C.
    max_frames: int | None = None

    # Used if the camera reports 0 fps
    fallback_fps: float = 30.0

    @classmethod
    def from_yaml(cls, config_path: Path) -> "Config":
        with open(config_path) as f:
            raw = yaml.safe_load(f) or {}

        # Log any unknown config keys
        unknown = set(raw) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"Unknown config keys: {sorted(unknown)}")

        # EasyPySpin reads an int as a camera index and a string as a serial
        # number, so an unquoted serial in the YAML would open the wrong camera.
        if raw.get("camera_serial") is not None:
            raw["camera_serial"] = str(raw["camera_serial"])

        # Paths in the config are relative to the config file itself
        base_dir = config_path.parent.resolve()
        for key in ("intrinsics", "output_dir"):
            if raw.get(key) is not None:
                raw[key] = (base_dir / Path(raw[key])).resolve()

        config = cls(**raw)
        if config.output_dir is None:
            config.output_dir = DEFAULT_OUTPUT_DIR
        return config


def require_spinnaker() -> None:
    """Fail with install instructions rather than a bare ImportError."""
    if IMPORT_ERROR is None:
        return
    raise SystemExit(
        f"Could not import the FLIR camera libraries ({IMPORT_ERROR}).\n"
        "  PySpin: install the Spinnaker SDK and its Python wheel from FLIR\n"
        "  EasyPySpin: pip install EasyPySpin"
    )


def list_cameras() -> None:
    """Print the serial number and model of every connected FLIR camera."""
    system = PySpin.System.GetInstance()
    try:
        cam_list = system.GetCameras()
        try:
            if cam_list.GetSize() == 0:
                print("No FLIR cameras found")
            for idx in range(cam_list.GetSize()):
                cam = cam_list.GetByIndex(idx)
                try:
                    # The transport-layer nodemap is readable without Init()
                    serial = cam.TLDevice.DeviceSerialNumber.GetValue()
                    model = cam.TLDevice.DeviceModelName.GetValue()
                    print(f"camera {idx}: serial {serial} ({model})")
                except PySpin.SpinnakerException as err:
                    print(f"camera {idx}: could not be queried ({err})")
                finally:
                    del cam
        finally:
            cam_list.Clear()
    finally:
        system.ReleaseInstance()


def load_camera(config: Config):
    """Load the intrinsics used to undistort corners, or None if not configured."""
    if config.intrinsics is None:
        print("No intrinsics given: centres assume a pinhole camera")
        return None

    # This script handles one camera. A caliscope .toml can hold a whole array,
    # so refuse a multi-camera file rather than silently picking one of them.
    cameras = CameraArray.from_toml(config.intrinsics).cameras
    if len(cameras) != 1:
        raise ValueError(
            f"{config.intrinsics} describes {len(cameras)} cameras "
            f"(ids {sorted(cameras)}); this script assumes a single camera."
        )
    cam_id, camera = next(iter(cameras.items()))
    print(f"Loaded intrinsics for camera {cam_id} from {config.intrinsics}")
    print(f"fisheye: {camera.fisheye}, calibrated at (w, h): {camera.size}")
    print(f"calibration RMSE (px): {camera.error}")

    # caliscope applies rotation_count 90-degree rotations to frames before
    # calibrating, so a nonzero value means these intrinsics describe a rotated
    # frame and would not match the raw frames we read here.
    if camera.rotation_count:
        raise ValueError(
            f"Camera {cam_id} was calibrated with "
            f"rotation_count={camera.rotation_count}; this script assumes "
            "unrotated frames."
        )
    return camera


def apply_setting(cap, prop: int, value, name: str) -> None:
    """Set one camera property, failing if the camera refuses it.

    Out-of-range values are clipped by EasyPySpin and still count as success, so
    a refusal here means the setting did not go through at all.
    """
    if not cap.set(prop, value):
        raise RuntimeError(f"Camera did not accept {name}={value}")


@dataclass(frozen=True)
class LiveControl:
    """One camera setting that the preview window can step with a key press.

    Named by its GenICam node rather than a cv2.CAP_PROP_*: EasyPySpin only
    maps a handful of properties to nodes, and BlackLevel is not among them, so
    going straight to the node keeps every control on one code path.

    The two keys are neighbours on the keyboard rather than a letter and its
    shifted self, because HighGUI does not report shift the same way everywhere.
    """

    node: str
    label: str  # shown in the preview overlay
    config_key: str  # the Config field holding this, for the end-of-run summary
    down_key: str
    up_key: str
    step: float  # what one key press changes the value by
    fmt: str = "{:.1f}"
    unit: str = ""
    # Nodes to set before writing this one, because otherwise the write does
    # nothing: an auto mode immediately overwrites the value, and the camera
    # ignores Gamma entirely while GammaEnable is false.
    prepare: tuple[tuple[str, object], ...] = ()


LIVE_CONTROLS = (
    LiveControl(
        node="Gain",
        label="gain",
        config_key="gain_db",
        down_key="g",
        up_key="h",
        step=1.0,
        unit=" dB",
        prepare=(("GainAuto", "Off"),),
    ),
    LiveControl(
        node="ExposureTime",
        label="exposure",
        config_key="exposure_us",
        down_key="e",
        up_key="r",
        step=1000.0,
        fmt="{:.0f}",
        unit=" us",
        prepare=(("ExposureAuto", "Off"),),
    ),
    LiveControl(
        node="Gamma",
        label="gamma",
        config_key="gamma",
        down_key="t",
        up_key="y",
        step=0.05,
        fmt="{:.2f}",
        prepare=(("GammaEnable", True),),
    ),
    LiveControl(
        node="BlackLevel",
        label="black",
        config_key="black_level",
        down_key="b",
        up_key="n",
        step=0.1,
        fmt="{:.2f}",
        unit=" %",
    ),
)


def read_node(cap, node: str) -> float | None:
    """Read a numeric camera node, or None if this camera does not have it.

    Missing nodes are a normal answer here rather than an error: not every FLIR
    model carries every one of these, and a control the camera lacks is one we
    simply do not offer.
    """
    try:
        value = cap.get_pyspin_value(node)
        return None if value is None else float(value)
    except (PySpin.SpinnakerException, TypeError, ValueError):
        return None


class LiveSettings:
    """The camera settings the preview can step, and their current values.

    Built by asking the camera which of LIVE_CONTROLS it actually has, so the
    keys on offer match the camera in front of you.

    Values are cached rather than read back every frame: a GenICam read per
    setting per frame is wasted work when only a key press changes them.
    """

    def __init__(self, cap):
        self.cap = cap
        self.controls = [c for c in LIVE_CONTROLS if read_node(cap, c.node) is not None]
        self.values = {c.node: read_node(cap, c.node) for c in self.controls}
        # key code -> (control, direction)
        self.keys = {}
        for control in self.controls:
            self.keys[ord(control.down_key)] = (control, -1)
            self.keys[ord(control.up_key)] = (control, +1)

    def handle_key(self, key: int) -> bool:
        """Step a setting if this key is bound to one; True if it was."""
        if key not in self.keys:
            return False
        control, direction = self.keys[key]

        for node, value in control.prepare:
            self.cap.set_pyspin_value(node, value)
        self.cap.set_pyspin_value(
            control.node, self.values[control.node] + direction * control.step
        )

        # Read back instead of trusting the write: the camera clips to its own
        # limits, so this is where "the gain will not go any higher" shows up.
        actual = read_node(self.cap, control.node)
        if actual is None or actual == self.values[control.node]:
            # Nothing moved: the setting is already at one of the camera's
            # limits, and repeating the same line for every further press of a
            # held key would bury the ones that did something.
            return True
        self.values[control.node] = actual
        print(f"{control.label} {self.format(control)}")
        return True

    def format(self, control: LiveControl) -> str:
        return control.fmt.format(self.values[control.node]) + control.unit

    def status(self) -> str:
        """One line for the preview overlay: value and keys for each setting."""
        return "  ".join(
            f"{c.label} {self.format(c)} [{c.down_key}/{c.up_key}]"
            for c in self.controls
        )

    def print_as_config(self) -> None:
        """Print the settings landed on, ready to paste into the YAML config."""
        if not self.controls:
            return
        print("\nCamera settings at the end of the run, for the config file:")
        for control in self.controls:
            value = control.fmt.format(self.values[control.node])
            print(f"  {control.config_key}: {value}")


def apply_node_setting(cap, node: str, value: float, name: str, prepare=()) -> None:
    """Set one camera node, failing if the camera refuses it.

    The node-name counterpart of apply_setting, for the settings EasyPySpin
    does not expose as a cv2 property.
    """
    for prep_node, prep_value in prepare:
        cap.set_pyspin_value(prep_node, prep_value)
    if not cap.set_pyspin_value(node, value):
        raise RuntimeError(f"Camera did not accept {name}={value}")


def open_camera(config: Config, camera):
    """Open the camera, apply the requested settings, and report the actual ones."""
    # int -> camera index, str -> serial number
    index = config.camera_serial if config.camera_serial is not None else 0
    cap = EasyPySpin.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open FLIR camera {index!r}. "
            "Run with --list-cameras to see what is connected."
        )

    if config.frame_width is not None:
        apply_setting(
            cap, cv2.CAP_PROP_FRAME_WIDTH, int(config.frame_width), "frame_width"
        )
    if config.frame_height is not None:
        apply_setting(
            cap, cv2.CAP_PROP_FRAME_HEIGHT, int(config.frame_height), "frame_height"
        )

    # Exposure before fps: the exposure time caps the frame rate, so setting it
    # first lets a short exposure make room for the requested rate.
    if config.exposure_us is not None:
        apply_setting(
            cap, cv2.CAP_PROP_EXPOSURE, float(config.exposure_us), "exposure_us"
        )
    if config.gain_db is not None:
        apply_setting(cap, cv2.CAP_PROP_GAIN, float(config.gain_db), "gain_db")

    # Not cv2 properties, so these go straight to the camera nodes -- the same
    # path the live preview keys use.
    if config.gamma is not None:
        apply_node_setting(
            cap,
            "Gamma",
            float(config.gamma),
            "gamma",
            prepare=(("GammaEnable", True),),
        )
    if config.black_level is not None:
        apply_node_setting(cap, "BlackLevel", float(config.black_level), "black_level")

    if config.fps is not None:
        apply_setting(cap, cv2.CAP_PROP_FPS, float(config.fps), "fps")

    # EasyPySpin always opens the camera in "NewestOnly", i.e. it skips whatever
    # piled up in the buffer, like a webcam. That is what we usually want here,
    # so this only has to act on the other case.
    if not config.buffer_newest_only:
        cap.cam.TLStream.StreamBufferHandlingMode.SetValue(
            PySpin.StreamBufferHandlingMode_OldestFirst
        )
    cap.grabTimeout = int(config.grab_timeout_ms)

    # Report what the camera actually ended up with
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # This is AcquisitionResultingFrameRate: the rate the camera can actually
    # sustain, which is lower than the requested one if the exposure is long.
    fps = cap.get(cv2.CAP_PROP_FPS) or config.fallback_fps

    # fx, fy, cx, cy are in pixels at the calibration resolution, so intrinsics
    # from a different resolution are silently wrong rather than obviously broken.
    if camera is not None and (width, height) != tuple(camera.size):
        cap.release()
        raise ValueError(
            f"Camera is {width}x{height} but the intrinsics were "
            f"calibrated at {camera.size[0]}x{camera.size[1]}."
        )

    print(
        f"Camera {cap.get_pyspin_value('DeviceModelName')} "
        f"(serial {cap.get_pyspin_value('DeviceSerialNumber')})"
    )
    print(
        f"Reading {width}x{height} @ {fps:.1f} fps, "
        f"exposure {cap.get(cv2.CAP_PROP_EXPOSURE) / 1000:.1f} ms, "
        f"gain {cap.get(cv2.CAP_PROP_GAIN):.1f} dB"
    )

    # Reported separately because not every camera has them, and gamma is not
    # in effect at all while GammaEnable is false.
    gamma = read_node(cap, "Gamma")
    if gamma is not None:
        disabled = "" if read_node(cap, "GammaEnable") else " (disabled)"
        print(f"gamma {gamma:.2f}{disabled}")
    black_level = read_node(cap, "BlackLevel")
    if black_level is not None:
        print(f"black level {black_level:.2f} %")

    return cap, (width, height), fps


def to_gray_and_bgr(frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split a raw camera frame into what the detector and the writer each want.

    These are mono cameras, so a frame arrives as a 2D array. STag detects on
    the grayscale, and sending that to the detector subprocess rather than a
    three-channel copy is a third of the pickling per frame; the BGR copy is
    what the annotations are drawn on and what the mp4v writer needs.

    Both are copies, which matters: the array from EasyPySpin is a view on a
    PySpin buffer that gets recycled.
    """
    if frame.dtype == np.uint16:
        frame = (frame >> 8).astype(np.uint8)  # e.g. a Mono16 pixel format
    elif frame.dtype != np.uint8:
        frame = cv2.convertScaleAbs(frame)

    if frame.ndim == 2:
        gray = frame.copy()
        return gray, cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    bgr = frame.copy()
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), bgr


def detect(config: Config, cap, camera, writer) -> tuple[list, int, list, float]:
    """Detect markers on live frames, annotating and writing as we go.

    Returns the (frame, marker_id, centre_x, centre_y) rows, the number of
    frames read, the indices of the frames the detector crashed on, and how
    long the run lasted in seconds.

    The preview's other keys step the camera settings in LIVE_CONTROLS, which
    take effect on the camera immediately: the next frame shows the result.

    Runs until 'q' in the preview, Ctrl-C, or `max_frames`. A camera that stops
    delivering ends the run the same way, so the frames already detected on are
    saved rather than thrown away with the traceback.
    """
    detector = StagDetector(libraryHD=config.stag_libraryHD)
    detections = []
    frame_idx = 0
    t_start = time.time()

    live = None
    if config.show_preview:
        # Resizable, because the full sensor is taller than a lot of screens
        cv2.namedWindow("markers", cv2.WINDOW_NORMAL)
        live = LiveSettings(cap)
        print(f"Preview keys: {live.status()}, [q] stop")

    try:
        while config.max_frames is None or frame_idx < config.max_frames:
            try:
                ok, raw = cap.read()
            except PySpin.SpinnakerException as err:
                # Most likely a grab timeout: a camera in trigger mode with
                # nothing triggering it looks exactly like this.
                print(f"\nCamera stopped delivering after {frame_idx} frames: {err}")
                break
            if not ok:
                print(f"\nIncomplete frame from the camera after {frame_idx} frames")
                break

            gray, image = to_gray_and_bgr(raw)

            # A frame that crashes the detector comes back as an empty
            # detection rather than taking the interpreter down with it.
            corners, ids, _ = detector.detect(gray)

            for marker_id, centre_x, centre_y in marker_centres(corners, ids, camera):
                detections.append((frame_idx, marker_id, centre_x, centre_y))
                # drawDetectedMarkers below prints the id at the corner mean,
                # which sits a little off this cross on a tilted marker -- that
                # gap is exactly the systematic offset marker_centre avoids.
                cv2.drawMarker(
                    image,
                    (int(round(centre_x)), int(round(centre_y))),
                    (0, 0, 255),
                    cv2.MARKER_CROSS,
                    markerSize=20,
                    thickness=2,
                )

            # the dot marks corner 0, the id is printed between corners 0 and 2
            stag.drawDetectedMarkers(image, corners, ids)

            if writer is not None:
                writer.write(image)

            if config.show_preview:
                # Drawn on a copy, so the counter does not end up in the video
                display = image.copy()
                cv2.putText(
                    display,
                    f"frame {frame_idx}  {len(ids)} markers  [q] stop",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )
                # Smaller: this line carries every setting and its keys
                cv2.putText(
                    display,
                    live.status(),
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )
                cv2.imshow("markers", display)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    frame_idx += 1
                    break
                # Anything else may be a setting to step
                live.handle_key(key)

            frame_idx += 1
    except KeyboardInterrupt:
        print(f"\nInterrupted after {frame_idx} frames")
    finally:
        detector.close()
        if live is not None:
            live.print_as_config()
        if config.show_preview:
            cv2.destroyAllWindows()
            for _ in range(4):  # nudge the window into actually closing
                cv2.waitKey(1)

    return detections, frame_idx, detector.skipped, time.time() - t_start


def save_centres(detections: list, centres_path: Path) -> None:
    np.savetxt(
        centres_path,
        np.array(detections, dtype=np.float64).reshape(-1, 4),
        delimiter=",",
        header="frame,marker_id,centre_x,centre_y",
        comments="",
        # frame and marker_id are integers, the centres are sub-pixel
        fmt=["%d", "%d", "%.4f", "%.4f"],
    )
    print(f"Saved {len(detections)} centres to {centres_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="Path to the YAML config file")
    parser.add_argument(
        "--list-cameras",
        action="store_true",
        help="List the connected FLIR cameras and their serial numbers, then exit",
    )
    args = parser.parse_args()

    require_spinnaker()

    if args.list_cameras:
        list_cameras()
        return

    config = Config.from_yaml(args.config)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    camera = load_camera(config)
    cap, size, fps = open_camera(config, camera)

    # Name the outputs after the camera: with two of them watching the same
    # scene, the serial is what tells the runs apart later.
    serial = cap.get_pyspin_value("DeviceSerialNumber") or "unknown"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    stem = f"flir{serial}_{timestamp}"
    video_path = config.output_dir / f"{stem}_markers.mp4"

    writer = None
    if config.save_annotated_video:
        writer = cv2.VideoWriter(
            str(video_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            size,
        )

    try:
        detections, n_frames, skipped, duration = detect(config, cap, camera, writer)
    finally:
        cap.release()
        if writer is not None:
            writer.release()

    # Crashed frames carry no detections, so they look just like empty frames in
    # the output. Report them: a high count means the results are sparse for a
    # reason that has nothing to do with the markers.
    print(f"\nRead {n_frames} frames in {duration:.1f} s")
    if n_frames:
        print(f"Detection ran at {n_frames / duration:.1f} fps")
    if skipped:
        print(
            f"{len(skipped)} frames were skipped after the stag detector crashed "
            f"on them (upstream bug, see crab_stags/stag_safe.py): {skipped}"
        )

    save_centres(detections, config.output_dir / f"{stem}_marker_centres.csv")
    if writer is not None:
        # The frames in it are the ones detection kept up with, not every frame
        # the camera sent, so it is written at the camera's rate but plays back
        # faster than real time whenever detection was the slower of the two.
        print(f"Saved annotated video to {video_path}")


if __name__ == "__main__":
    main()
