"""Detect STag markers in a video or live camera and log their centres.

Driven by a YAML config file (see `configs/detect_markers_config.yaml`):

    python scripts/detect_markers.py configs/detect_markers_config.yaml

Writes one CSV row per detected marker per frame (frame, marker_id, centre_x,
centre_y), and optionally an annotated .mp4 alongside it. Both go to the repo's
output/ folder unless the config overrides `output_dir`.

The centre is the intersection of the marker's image diagonals rather than the
mean of its corners; see `crab_stags/markers.py` for why. Detection runs in a
subprocess because stag-python segfaults on some real frames; see
`crab_stags/stag_safe.py`.
"""

import argparse
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import stag
import yaml
from caliscope.api import CameraArray

from crab_stags.markers import marker_centres
from crab_stags.stag_safe import StagDetector

# Repo-level output folder, a sibling of scripts/
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"


@dataclass
class Config:
    """Inputs for a marker detection run."""

    # An int is a camera index (0 is usually the built-in webcam);
    # a string is a path to a video file.
    video_source: int | Path = 0

    # Camera intrinsics from scripts/compute_intrinsics.py. Optional but recommended,
    # used to undistort the marker corners before the centre is computed.
    # Must describe a single camera (see load_camera).
    intrinsics: Path | None = None

    # STag library/family: one of [11, 13, 15, 17, 19, 21, 23]
    # see https://github.com/ManfredStoiber/stag-python#-configuration
    stag_libraryHD: int = 15

    # Outputs
    # Defaults to DEFAULT_OUTPUT_DIR
    output_dir: Path | None = None
    save_annotated_video: bool = True

    # Preview
    # press q to stop
    show_preview: bool = False

    # Used if the source reports 0 fps (webcams often do)
    fallback_fps: float = 30.0

    @classmethod
    def from_yaml(cls, config_path: Path) -> "Config":
        with open(config_path) as f:
            raw = yaml.safe_load(f) or {}

        # Log any unknown config keys
        unknown = set(raw) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"Unknown config keys: {sorted(unknown)}")

        # Paths in the config are relative to the config file itself
        base_dir = config_path.parent.resolve()
        for key in ("intrinsics", "output_dir"):
            if raw.get(key) is not None:
                raw[key] = (base_dir / Path(raw[key])).resolve()
        # A string video_source is a path, an int is a camera index
        if isinstance(raw.get("video_source"), str):
            raw["video_source"] = (base_dir / Path(raw["video_source"])).resolve()

        config = cls(**raw)
        if config.output_dir is None:
            config.output_dir = DEFAULT_OUTPUT_DIR
        return config


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


def open_video(config: Config, camera):
    """Open the video source and report the frame size and fps to use."""
    source = (
        str(config.video_source)
        if isinstance(config.video_source, Path)
        else config.video_source
    )
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {config.video_source}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or np.isnan(fps) or fps <= 0:
        fps = config.fallback_fps

    # fx, fy, cx, cy are in pixels at the calibration resolution, so intrinsics
    # from a different resolution are silently wrong rather than obviously broken.
    if camera is not None and (width, height) != tuple(camera.size):
        cap.release()
        raise ValueError(
            f"Video source is {width}x{height} but the intrinsics were "
            f"calibrated at {camera.size[0]}x{camera.size[1]}."
        )

    print(f"Reading {width}x{height} @ {fps:.1f} fps")
    return cap, (width, height), fps


def detect(config: Config, cap, camera, writer) -> tuple[list, int, list]:
    """Detect markers frame by frame, annotating and writing as we go.

    Returns the (frame, marker_id, centre_x, centre_y) rows, the number of
    frames read, and the indices of the frames the detector crashed on.
    """
    detector = StagDetector(libraryHD=config.stag_libraryHD)
    detections = []
    frame_idx = 0

    try:
        while True:
            ok, image = cap.read()
            if not ok:
                # end of file, or dropped camera
                break

            # A frame that crashes the detector comes back as an empty
            # detection rather than taking the interpreter down with it.
            corners, ids, _ = detector.detect(image)

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
                cv2.imshow("markers", image)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_idx += 1
    finally:
        detector.close()

    return detections, frame_idx, detector.skipped


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
    args = parser.parse_args()

    config = Config.from_yaml(args.config)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    stem = (
        config.video_source.stem
        if isinstance(config.video_source, Path)
        else f"camera_{config.video_source}"
    )

    camera = load_camera(config)
    cap, size, fps = open_video(config, camera)

    writer = None
    if config.save_annotated_video:
        writer = cv2.VideoWriter(
            str(config.output_dir / f"{stem}_markers.mp4"),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            size,
        )

    try:
        detections, n_frames, skipped = detect(config, cap, camera, writer)
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if config.show_preview:
            cv2.destroyAllWindows()
            for _ in range(4):  # nudge macOS into actually closing the window
                cv2.waitKey(1)

    # Crashed frames carry no detections, so they look just like empty frames in
    # the output. Report them: a high count means the results are sparse for a
    # reason that has nothing to do with the markers.
    print(f"\nRead {n_frames} frames")
    if skipped:
        print(
            f"{len(skipped)} frames were skipped after the stag detector crashed "
            f"on them (upstream bug, see crab_stags/stag_safe.py): {skipped}"
        )

    save_centres(detections, config.output_dir / f"{stem}_marker_centres.csv")
    if writer is not None:
        print(f"Saved annotated video to {config.output_dir / f'{stem}_markers.mp4'}")


if __name__ == "__main__":
    main()
