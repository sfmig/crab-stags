"""Record a video from a FLIR (Spinnaker) camera connected to this machine.

The FLIR sibling of `scripts/capture_calibration_video.py`: OpenCV's
VideoCapture cannot open a machine-vision camera, so this goes through
EasyPySpin (https://github.com/elerac/EasyPySpin), a thin wrapper that gives
FLIR's PySpin a cv2.VideoCapture-like interface.

Needs the Spinnaker SDK and its PySpin wheel (both from FLIR, not on PyPI),
then:
    pip install EasyPySpin

Driven by a YAML config file (see `configs/record_flir_camera_config.yaml`):

    python scripts/capture_calibration_video_flir.py configs/record_flir_camera_config.yaml

The camera is taken as it is set up: this script does not touch trigger mode
or any other persistent camera state beyond the settings named in the config.
A camera in trigger mode therefore delivers nothing unless something triggers
it, and the run fails with a PySpin grab timeout instead of quietly rewiring it.

Shows a live preview: press 's' to start recording, 'q' to stop and quit.
Use this to collect a calibration video (e.g. moving a charuco board around
the field of view), then feed the resulting .mp4 to
`scripts/compute_intrinsics.py`.

To list the connected cameras and their serial numbers:
    python scripts/capture_calibration_video_flir.py configs/record_flir_camera_config.yaml --list-cameras
"""

import argparse
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import yaml

try:
    import EasyPySpin
    import PySpin
except ImportError as err:  # so --help still works without the SDK installed
    EasyPySpin = PySpin = None
    IMPORT_ERROR: ImportError | None = err
else:
    IMPORT_ERROR = None


@dataclass
class Config:
    """Inputs for a FLIR camera recording run."""

    # Serial number of the camera to open, as a string (run with
    # --list-cameras to see them). None takes the first camera found.
    camera_serial: str | None = None

    output_dir: Path | None = None

    # Requested capture settings -- None keeps whatever the camera currently has.
    # EasyPySpin clips out-of-range values, so we use what it reports back.
    frame_width: int | None = None
    frame_height: int | None = None
    fps: float | None = None
    exposure_us: float | None = None
    gain_db: float | None = None

    # False keeps every frame the camera delivers (the preview falls behind if
    # the writer cannot keep up); True always shows the newest one instead.
    buffer_newest_only: bool = False

    # How long to wait for a frame before giving up
    grab_timeout_ms: int = 5000

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
        if raw.get("output_dir") is not None:
            raw["output_dir"] = (
                config_path.parent.resolve() / Path(raw["output_dir"])
            ).resolve()

        config = cls(**raw)
        if config.output_dir is None:
            config.output_dir = config_path.parent.resolve()
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


def apply_setting(cap, prop: int, value, name: str) -> None:
    """Set one camera property, failing if the camera refuses it.

    Out-of-range values are clipped by EasyPySpin and still count as success, so
    a refusal here means the setting did not go through at all.
    """
    if not cap.set(prop, value):
        raise RuntimeError(f"Camera did not accept {name}={value}")


def open_camera(config: Config):
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
    if config.fps is not None:
        apply_setting(cap, cv2.CAP_PROP_FPS, float(config.fps), "fps")

    # EasyPySpin always opens the camera in "NewestOnly", i.e. it skips whatever
    # piled up in the buffer, like a webcam. For a recording we usually want
    # every frame instead, even if that means the preview lags.
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
    print(
        f"Camera {cap.get_pyspin_value('DeviceModelName')} "
        f"(serial {cap.get_pyspin_value('DeviceSerialNumber')})"
    )
    print(
        f"Recording at {width}x{height} @ {fps:.1f} fps, "
        f"exposure {cap.get(cv2.CAP_PROP_EXPOSURE) / 1000:.1f} ms, "
        f"gain {cap.get(cv2.CAP_PROP_GAIN):.1f} dB"
    )

    return cap, (width, height), fps


def to_bgr(frame: np.ndarray) -> np.ndarray:
    """Convert a raw camera frame to the 8-bit BGR the video writer expects.

    These are mono cameras, so a frame arrives as a 2D array while the mp4v
    writer wants three channels. This also copies the data, which matters: the
    array from EasyPySpin is a view on a PySpin buffer that gets recycled.
    """
    if frame.dtype == np.uint16:
        frame = (frame >> 8).astype(np.uint8)  # e.g. a Mono16 pixel format
    elif frame.dtype != np.uint8:
        frame = cv2.convertScaleAbs(frame)

    if frame.ndim == 2:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    return frame.copy()


def record(cap, size, fps, output_path: Path) -> None:
    """Preview the camera; 's' starts recording, 'q' stops and closes the window."""
    writer = None
    recording = False
    n_frames = 0
    t_start = None

    # Resizable, because the full sensor is taller than a lot of screens
    cv2.namedWindow("camera", cv2.WINDOW_NORMAL)

    try:
        while True:
            # Get frame. A camera that is not delivering -- in trigger mode with
            # nothing triggering it, say -- raises a PySpin grab timeout here
            # after grab_timeout_ms, which is left to propagate.
            ok, raw = cap.read()
            if not ok:
                raise RuntimeError(
                    f"Camera returned an incomplete frame after {n_frames} frames"
                )
            frame = to_bgr(raw)

            # If recording: save frame to video output (without the overlay)
            if recording:
                writer.write(frame)
                n_frames += 1

            # Draw the status text on a frame copy, so it doesn't end up in the video
            display = frame.copy()
            if recording:
                label = f"RECORDING  {n_frames} frames  [q] stop"
                colour = (0, 0, 255)
            else:
                label = "preview  [s] start recording  [q] quit"
                colour = (0, 255, 0)
            cv2.putText(
                display,
                label,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                colour,
                2,
            )
            # show in a window called "camera"
            cv2.imshow("camera", display)

            # Get and parse key strokes
            key = cv2.waitKey(1) & 0xFF
            if key == ord("s") and not recording:
                # intialise writer
                writer = cv2.VideoWriter(
                    str(output_path),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    fps,
                    size,
                )
                recording = True
                t_start = time.time()
                print("Recording started")
            elif key == ord("q"):
                break
    finally:
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()
        for _ in range(4):  # nudge the window into actually closing
            cv2.waitKey(1)

    if not n_frames:
        print("Nothing recorded")
        return

    duration = time.time() - t_start
    print(f"Saved {n_frames} frames ({duration:.1f} s) to {output_path}")
    print(f"Measured fps: {n_frames / duration:.1f} (written to file as {fps:.1f})")


def report(output_path: Path) -> None:
    """Sanity check: read the saved video back."""
    cap = cv2.VideoCapture(str(output_path))
    print("\n--- Saved video ---")
    print(f"frames: {int(cap.get(cv2.CAP_PROP_FRAME_COUNT))}")
    print(f"fps: {cap.get(cv2.CAP_PROP_FPS)}")
    print(
        f"size: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}"
        f"x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}"
    )
    cap.release()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "config",
        type=Path,
        help="Path to the YAML config file",
    )
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

    cap, size, fps = open_camera(config)

    # Name the file after the camera: with two of them recording the same
    # board, the serial is what tells the videos apart later.
    serial = cap.get_pyspin_value("DeviceSerialNumber") or "unknown"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    output_path = config.output_dir / f"calibration_video_flir{serial}_{timestamp}.mp4"

    try:
        record(cap, size, fps, output_path)
    finally:
        cap.release()

    if output_path.exists():
        report(output_path)


if __name__ == "__main__":
    main()
