"""Detect STag markers in a video (or live camera) and log their centres.

Driven by a YAML config file (see `configs/detect_markers_config.yaml`):

    python scripts/detect_markers.py configs/detect_markers_config.yaml

For every frame it writes one CSV row per detected marker: the frame index, the
marker id, and the projection of the marker's centre, in pixels of the raw
(still distorted) image.

Set `intrinsics` in the config to make that centre exact; without it the script
assumes a pinhole camera and is off by up to about a pixel near the frame edge
(see marker_centre). For the marker's 3D pose, see notebooks/notebook_stags.py.
"""

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import stag
import yaml
from caliscope.api import CameraArray

# project_points dispatches between the Brown-Conrady and fisheye distortion
# models. Not part of caliscope.api's public surface, but it is the same helper
# the library uses internally for all of its own reprojection.
from caliscope.core.reprojection import project_points

# stag.detectMarkers still segfaults on some real frames (upstream buffer
# overrun, unfixed in stag-python 1.1.1), so detection runs in a subprocess.
# Put the repo root on the path so crab_stags imports no matter where this
# script is run from.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from crab_stags.stag_safe import StagDetector  # noqa: E402


@dataclass
class Config:
    """Inputs for a marker detection run."""

    # An int is a camera index (0 = default webcam); a string is a video file
    # path, resolved relative to this config file.
    video_source: int | Path = 0

    # STag library/family: one of [11, 13, 15, 17, 19, 21, 23]
    # https://github.com/manfredstoiber/stag-python#-configuration
    stag_libraryHD: int = 15

    # Camera intrinsics .toml from scripts/compute_intrinsics.py, resolved
    # relative to this config file. Optional: without it the marker centre is
    # computed assuming a pinhole camera (see marker_centre).
    intrinsics: Path | None = None
    cam_id: int = 0

    # Where the .csv (and the annotated .mp4, if any) go
    output_dir: Path | None = None

    # Write a copy of the video with the detections drawn on it
    save_annotated_video: bool = True

    # Show a live preview window (press q to stop)
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
        if not isinstance(raw.get("video_source", 0), int):
            raw["video_source"] = (base_dir / Path(raw["video_source"])).resolve()
            if not raw["video_source"].exists():
                raise FileNotFoundError(f"Video not found: {raw['video_source']}")
        if raw.get("intrinsics") is not None:
            raw["intrinsics"] = (base_dir / Path(raw["intrinsics"])).resolve()
            if not raw["intrinsics"].exists():
                raise FileNotFoundError(f"Intrinsics not found: {raw['intrinsics']}")
        if raw.get("output_dir") is not None:
            raw["output_dir"] = (base_dir / Path(raw["output_dir"])).resolve()

        config = cls(**raw)
        if config.output_dir is None:
            config.output_dir = (
                config.video_source.parent
                if isinstance(config.video_source, Path)
                else base_dir
            )
        return config


def intersect_diagonals(corners):
    """Intersection of the quadrilateral's two diagonals, in the given 2D frame.

    Both diagonals are lines through two known points, so in homogeneous
    coordinates each is a cross product of its endpoints, and their
    intersection is the cross product of the two lines.
    """
    h = np.hstack([corners, np.ones((4, 1))])
    point = np.cross(np.cross(h[0], h[2]), np.cross(h[1], h[3]))
    if abs(point[2]) < 1e-12:
        # The diagonals came out parallel, i.e. the intersection is at infinity.
        # Impossible for the diagonals of a convex quadrilateral, so this means
        # a degenerate detection rather than a marker; let the caller drop it.
        return None
    return point[:2] / point[2]


def marker_centre(marker_corners, camera=None) -> tuple[float, float] | None:
    """Projection of the marker's centre, in pixels of the raw image.

    Taken as the intersection of the image diagonals, NOT the mean of the four
    corners. Perspective preserves cross-ratio rather than ratios along a line,
    so the image of a square is a general quadrilateral whose vertex mean sits
    off the true centre -- for a 10 cm marker tilted 45 degrees against these
    intrinsics, by 0.5 px at 2 m, 2 px at 1 m and 8 px at 0.5 m. A systematic
    offset, not noise. What perspective does preserve is incidence, and the
    centre of a square is where its diagonals cross, so that construction
    survives the projection exactly.

    Exactly, that is, for a pinhole camera. A real lens bends the diagonals
    into curves, and intersecting them as straight lines is then off by up to
    about a pixel near the frame edge. So when `camera` is given we undistort
    the corners into normalized coordinates first (a true pinhole plane), take
    the intersection there, and project that point back out through the full
    lens model, which is exact. Without `camera` the corners are used as they
    come and the pinhole assumption stands.
    """
    corners = np.asarray(marker_corners, dtype=np.float64).reshape(4, 2)

    if camera is None:
        centre = intersect_diagonals(corners)
        return None if centre is None else (float(centre[0]), float(centre[1]))

    # Normalized coordinates are what the camera would have seen with no lens
    # distortion and no intrinsics applied, so the diagonals are straight there.
    centre_norm = intersect_diagonals(
        np.asarray(camera.undistort_points(corners, output="normalized"), dtype=np.float64).reshape(4, 2)
    )
    if centre_norm is None:
        return None

    # Send it back to pixels the way caliscope projects anything else: treat the
    # normalized point as the 3D ray (x, y, 1) and project it with no further
    # rotation or translation. project_points dispatches on the lens model, so
    # this is right for both Brown-Conrady and fisheye coefficients.
    ray = np.array([[centre_norm[0], centre_norm[1], 1.0]], dtype=np.float64)
    pixel = project_points(
        ray,
        np.zeros(3),
        np.zeros(3),
        camera.matrix,
        camera.distortions,
        camera.fisheye,
    ).reshape(2)
    return float(pixel[0]), float(pixel[1])


def load_camera(config: Config):
    """Load the calibrated camera, or None if the config gives no intrinsics."""
    if config.intrinsics is None:
        return None
    camera = CameraArray.from_toml(config.intrinsics)[config.cam_id]

    # caliscope applies rotation_count 90-degree rotations to frames before
    # calibrating, so a nonzero value means these intrinsics describe a rotated
    # frame and would not match the raw frames read here.
    if camera.rotation_count:
        raise ValueError(
            f"Camera {config.cam_id} was calibrated with rotation_count="
            f"{camera.rotation_count}; this script assumes unrotated frames."
        )
    return camera


def run(config: Config, output_stem: str):
    """Read the source frame by frame, detect markers, and write the CSV."""
    camera = load_camera(config)

    cap = cv2.VideoCapture(
        config.video_source
        if isinstance(config.video_source, int)
        else str(config.video_source)
    )
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {config.video_source}")

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or np.isnan(fps) or fps <= 0:
        fps = config.fallback_fps

    # fx, fy, cx, cy and the distortion coefficients are tied to the resolution
    # they were calibrated at, so intrinsics from another resolution would be
    # silently wrong rather than obviously broken.
    if camera is not None and (frame_w, frame_h) != tuple(camera.size):
        cap.release()
        raise ValueError(
            f"Video source is {frame_w}x{frame_h} but the intrinsics were "
            f"calibrated at {camera.size[0]}x{camera.size[1]}."
        )

    writer = None
    if config.save_annotated_video:
        writer = cv2.VideoWriter(
            str(config.output_dir / f"{output_stem}_detections.mp4"),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (frame_w, frame_h),
        )

    csv_path = config.output_dir / f"{output_stem}_centres.csv"
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["frame", "marker_id", "centre_x", "centre_y"])

    # Detection is isolated in a subprocess, so a frame that crashes the
    # detector costs us that frame rather than the whole run. The lost frames
    # are listed in detector.skipped and reported at the end.
    # verbose=False: crashes are common enough on some footage that a line per
    # crash drowns everything else; the count is reported at the end instead.
    detector = StagDetector(libraryHD=config.stag_libraryHD, verbose=False)

    frame_idx = 0
    n_detections = 0
    try:
        while True:
            ret, image = cap.read()
            if not ret:
                # end of file, or dropped camera
                break

            # Same signature and return as stag.detectMarkers
            corners, ids, _ = detector.detect(image)

            if ids is not None and len(ids) > 0:
                for marker_corners, marker_id in zip(corners, ids.flatten()):
                    centre = marker_centre(marker_corners, camera)
                    if centre is None:
                        # Degenerate quadrilateral; no centre to report.
                        continue
                    x, y = centre
                    csv_writer.writerow([frame_idx, int(marker_id), x, y])
                    n_detections += 1

                    if writer is not None or config.show_preview:
                        cv2.drawMarker(
                            image,
                            (int(round(x)), int(round(y))),
                            (0, 0, 255),
                            cv2.MARKER_CROSS,
                            markerSize=20,
                            thickness=2,
                        )

            if writer is not None or config.show_preview:
                # the dot marks corner 0, the id is printed midway between
                # corners 0 and 2
                stag.drawDetectedMarkers(image, corners, ids)

            if writer is not None:
                writer.write(image)

            if config.show_preview:
                cv2.imshow("detections", image)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_idx += 1

    # A finally block runs no matter how the "try" block ends
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if config.show_preview:
            cv2.destroyAllWindows()
        detector.close()
        csv_file.close()

    return csv_path, frame_idx, n_detections, detector.skipped


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="Path to the YAML config file")
    parser.add_argument(
        "--show", action="store_true", help="Show a live preview window (press q to stop)",
    )
    args = parser.parse_args()

    config = Config.from_yaml(args.config)
    if args.show:
        config.show_preview = True
    config.output_dir.mkdir(parents=True, exist_ok=True)

    if config.intrinsics is None:
        print(
            "No intrinsics given: marker centres assume a pinhole camera, so "
            "they are off by up to ~1 px near the frame edge."
        )
    else:
        print(f"Undistorting marker corners with {config.intrinsics}")

    stem = (
        config.video_source.stem
        if isinstance(config.video_source, Path)
        else f"camera{config.video_source}"
    )
    csv_path, n_frames, n_detections, skipped = run(config, stem)

    print(f"\nProcessed {n_frames} frames, {n_detections} marker detections")
    print(f"Saved marker centres to {csv_path}")

    # Frames the detector crashed on carry no detections, so they look exactly
    # like frames with no marker in them. Report them: a high count means the
    # track is sparse for a reason that has nothing to do with the markers.
    if skipped:
        print(
            f"\n{len(skipped)} of {n_frames} frames were skipped after the stag "
            f"detector crashed on them (upstream bug, see crab_stags/stag_safe.py)."
        )
        preview = skipped[:20]
        more = "" if len(skipped) == len(preview) else " ..."
        print(f"skipped frame indices: {preview}{more}")


if __name__ == "__main__":
    main()
