"""Record a video from a camera connected to this machine.

Driven by a YAML config file (see `configs/record_camera_config.yaml`):

    python scripts/capture_calibration_video.py configs/record_camera_config.yaml

Shows a live preview: press 's' to start recording, 'q' to stop and quit.
Use this to collect a calibration video (e.g. moving a charuco board around
the field of view), then feed the resulting .mp4 to
`scripts/compute_intrinsics.py`.

To list available cameras:
    python scripts/capture_calibration_video.py configs/record_camera_config.yaml --list-cameras

macOS note: the first run will ask for camera permission for the app running
this script (VSCode / Terminal). If no window appears, check
System Settings > Privacy & Security > Camera.
"""

import argparse
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import yaml


@dataclass
class Config:
    """Inputs for a camera recording run."""

    # 0 is usually the built-in webcam
    camera_index: int = 0

    output_dir: Path | None = None

    # Requested capture settings -- None keeps whatever the camera defaults to.
    # The camera may not honour them, so we use what it actually reports back.
    frame_width: int | None = None
    frame_height: int | None = None
    fps: float | None = None

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

        # Paths in the config are relative to the config file itself
        if raw.get("output_dir") is not None:
            raw["output_dir"] = (
                config_path.parent.resolve() / Path(raw["output_dir"])
            ).resolve()

        config = cls(**raw)
        if config.output_dir is None:
            config.output_dir = config_path.parent.resolve()
        return config


def capture_backend() -> int:
    """The native capture backend for this platform.

    CAP_ANY would let OpenCV choose, but naming the backend avoids it silently
    falling back to a slow or broken one.
    """
    if sys.platform == "darwin":
        return cv2.CAP_AVFOUNDATION
    if sys.platform == "win32":
        return cv2.CAP_DSHOW
    return cv2.CAP_V4L2


def list_cameras(n_indices: int = 4) -> None:
    """Print which camera indices can be opened.

    A bit slow, each failed index takes a moment to time out.
    """
    for idx in range(n_indices):
        cap = cv2.VideoCapture(idx, capture_backend())
        if not cap.isOpened():
            print(f"camera {idx}: not available")
        else:
            ok, frame = cap.read()
            if ok:
                print(f"camera {idx}: available, frame shape {frame.shape}")
            else:
                print(f"camera {idx}: opened but returned no frame")
        cap.release()


def open_camera(config: Config):
    """Open the camera, apply the requested settings, and report the actual ones."""
    cap = cv2.VideoCapture(config.camera_index, capture_backend())
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera {config.camera_index}")

    # Set image size and fps if defined
    if config.frame_width is not None:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.frame_width)
    if config.frame_height is not None:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.frame_height)
    if config.fps is not None:
        cap.set(cv2.CAP_PROP_FPS, config.fps)

    # Report actual img size and fps
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or config.fallback_fps
    print(f"Recording at {width}x{height} @ {fps:.1f} fps")

    return cap, (width, height), fps


def record(cap, size, fps, output_path: Path) -> None:
    """Preview the camera; 's' starts recording, 'q' stops and closes the window."""
    writer = None
    recording = False
    n_frames = 0
    t_start = None

    try:
        while True:
            # Get frame
            ok, frame = cap.read()
            if not ok:
                print("Dropped frame from camera, stopping")
                break

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
        for _ in range(4):  # nudge macOS into actually closing the window
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
        help="List the available camera indices and exit",
    )
    args = parser.parse_args()

    if args.list_cameras:
        list_cameras()
        return

    config = Config.from_yaml(args.config)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    output_path = config.output_dir / f"calibration_video_{timestamp}.mp4"

    cap, size, fps = open_camera(config)
    try:
        record(cap, size, fps, output_path)
    finally:
        cap.release()

    if output_path.exists():
        report(output_path)


if __name__ == "__main__":
    main()
