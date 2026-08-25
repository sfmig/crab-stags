"""Single-camera intrinsic calibration from a charuco video.

Driven by a YAML config file (see `configs/intrinsic_calibration_config.yaml`)
plus the video to calibrate from:

    python scripts/compute_intrinsics.py configs/intrinsic_calibration_config.yaml \
        calibration_video_20260807_1823.mp4

The video is looked up in the config's `calibration_dir` unless it is given as a
path that exists as-is (absolute, or relative to the current directory).

Writes a caliscope camera-array TOML with the estimated intrinsics, and
(optionally) a set of diagnostic plots next to it.

Reference: https://github.com/mprib/caliscope/blob/main/scripts/demo_api.py
"""

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import yaml
from caliscope.api import (
    CameraArray,
    Charuco,
    CharucoTracker,
    calibrate_intrinsics,
    extract_image_points,
)

# Grid used by caliscope's frame selector to score coverage of the image
COVERAGE_GRID = 5

# Minimum distinct board tilt directions, out of 8, for the focal length to be
# observable (Zhang 2000); this is min_orientations in select_calibration_frames
MIN_ORIENTATIONS = 4


def _resolve_path(value: str | None, base_dir: Path) -> Path | None:
    """Resolve a config path against `base_dir`, keeping None as None."""
    return None if value is None else (base_dir / Path(value)).resolve()


@dataclass
class Config:
    """Inputs for a single-camera intrinsic calibration run."""

    # Charuco board geometry.
    # NOTE: the square size does not affect the intrinsics -- intrinsic
    # calibration is scale-free, so you can measure the board later or use a
    # differently sized board for the extrinsic stage.
    charuco_n_cols: int
    charuco_n_rows: int
    charuco_square_sz_cm: float = 1.0

    # Fisheye lenses use 4 distortion coefficients (k1, k2, k3, k4) and assumes an
    # equidistant model; standard lense uses 5 coefficient (k1, k2, p1, p2, k3).

    # See: https://github.com/mprib/caliscope/blob/ddda95b44ba281c9bf968d2d0acbf7b0ab167e7d/src/caliscope/core/reprojection.py#L21
    # ATT! cv2.solvePnP interprets a length-4 distCoeffs as (k1, k2, p1, p2)
    # in the plumb-bob/Brown-Conrady/standard model. So we need to pass undistorted points.
    # (solvePnP has no fisheye mode)
    is_camera_fisheye: bool = False

    # Use every Nth frame for 2D landmark extraction
    frame_step: int = 1

    # Where the calibration videos live (defaults to the config file's folder)
    calibration_dir: Path | None = None

    output_dir: Path | None = None

    # diagnostic plots
    make_plots: bool = True

    # The video to calibrate from -- given on the command line, not in the YAML
    calibration_video: Path = field(init=False)

    @classmethod
    def from_yaml(cls, config_path: Path, video: Path) -> "Config":
        """Instantiate a config object from a yaml file."""
        with open(config_path) as f:
            raw_dict = yaml.safe_load(f) or {}

        if "calibration_video" in raw_dict:
            raise ValueError(
                "calibration_video is now a command-line argument, not a config "
                "key; set calibration_dir instead"
            )

        # Log any unknown config keys
        unknown = set(raw_dict) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"Unknown config keys: {sorted(unknown)}")

        # Resolve config paths
        # Paths in the config are relative to the config file itself
        base_dir = config_path.parent.resolve()
        for key in ("calibration_dir", "output_dir"):
            if key in raw_dict:
                raw_dict[key] = _resolve_path(raw_dict[key], base_dir)

        # Instantiate a config object
        config = cls(**raw_dict)

        # If calibration dir is None (unset), set to config parent dir  
        if config.calibration_dir is None:
            config.calibration_dir = base_dir

        # Resolve path to calibration video from CLI args and config
        # A path that exists as given wins; otherwise look the video up in
        # calibration_dir
        config.calibration_video = (
            video.resolve()
            if video.exists()
            else (config.calibration_dir / video).resolve()
        )
        if not config.calibration_video.exists():
            raise FileNotFoundError(
                f"Calibration video not found: {config.calibration_video}"
            )

        # Set output dir to the dir where calibration video is if unset
        if config.output_dir is None:
            config.output_dir = config.calibration_video.parent
        return config


def run_calibration(config: Config):
    """Extract charuco corners from the video and calibrate the camera."""
    # Initialise charuco tracker
    charuco = Charuco.from_squares(
        columns=config.charuco_n_cols,
        rows=config.charuco_n_rows,
        square_size_cm=config.charuco_square_sz_cm,
    )
    tracker = CharucoTracker(charuco)

    # Initialise camera (pixel resolution is read from the video metadata)
    cameras = CameraArray.from_video_metadata({0: config.calibration_video})
    cameras[0].fisheye = config.is_camera_fisheye

    # Extract 2d landmarks from calibration video
    points = extract_image_points(
        config.calibration_video,
        cam_id=0,
        tracker=tracker,
        frame_step=config.frame_step,
    )

    # NOTE: at most the best ~30 frames are used (grid_count in the .toml file)
    output = calibrate_intrinsics(points, cameras[0])
    cameras[0] = output.camera

    return cameras, points, output


def report(cameras, output) -> None:
    camera, rep = cameras[0], output.report
    print("\n--- Intrinsics ---")
    print(f"matrix:\n{camera.matrix}")
    print(f"distortions: {camera.distortions}")
    print(f"fisheye: {camera.fisheye}")
    print(f"image size (px): {camera.size}")
    verdict, checks, actions = assess_quality(rep)
    print("\n--- Quality ---")
    print(f"VERDICT: {verdict}\n")
    for line in checks:
        print(f"  {line}")

    if actions:
        print("\n--- What to do next ---")
        for i, action in enumerate(actions, start=1):
            for line in _wrap(f"{i}. {action}", width=76, indent="   "):
                print(f"  {line}")


# ----------------------------------------------------------------------------
# Diagnostics
# ----------------------------------------------------------------------------


def _frame_arrays(points, selected_frames):
    """Per-frame (sync_index, object points, image points) for the used frames."""
    df = points.df
    per_frame = []
    for sync_index in selected_frames:
        frame_df = df[df["sync_index"] == sync_index]
        if len(frame_df) < 4:
            continue
        obj = np.nan_to_num(
            np.asarray(
                frame_df[["obj_loc_x", "obj_loc_y", "obj_loc_z"]], dtype=np.float64
            )
        )
        img = np.asarray(frame_df[["img_loc_x", "img_loc_y"]], dtype=np.float64)
        per_frame.append((sync_index, obj, img))
    return per_frame


def _reprojection_residuals(camera, per_frame):
    """Reproject each frame's board corners and return (sync_index, residuals) pairs.

    Residuals are (N, 2) arrays of (projected - observed) image coordinates, in px.
    """
    matrix = np.asarray(camera.matrix, dtype=np.float64)
    dist = np.asarray(camera.distortions, dtype=np.float64).ravel()
    results = []

    for sync_index, obj, img in per_frame:
        obj32 = obj.astype(np.float32)
        img32 = img.astype(np.float32)

        if camera.fisheye:
            # solvePnP has no fisheye mode: undistort to normalised rays first,
            # then solve with an identity camera and no distortion.
            undistorted = cv2.fisheye.undistortPoints(
                img32.reshape(-1, 1, 2), matrix, dist
            )
            ok, rvec, tvec = cv2.solvePnP(
                obj32, undistorted, np.eye(3), np.zeros(4), flags=cv2.SOLVEPNP_ITERATIVE
            )
            if not ok:
                continue
            projected, _ = cv2.fisheye.projectPoints(
                obj32.reshape(-1, 1, 3), rvec, tvec, matrix, dist
            )
        else:
            ok, rvec, tvec = cv2.solvePnP(
                obj32, img32, matrix, dist, flags=cv2.SOLVEPNP_ITERATIVE
            )
            if not ok:
                continue
            projected, _ = cv2.projectPoints(obj32, rvec, tvec, matrix, dist)

        results.append((sync_index, projected.reshape(-1, 2) - img))

    return results


def plot_corner_coverage(points, report_, camera, ax) -> None:
    """Where in the image the board was seen -- gaps here mean shaky intrinsics."""
    width, height = camera.size
    df = points.df
    selected = df["sync_index"].isin(report_.selected_frames)

    ax.scatter(
        df.loc[~selected, "img_loc_x"],
        df.loc[~selected, "img_loc_y"],
        s=2,
        c="0.8",
        label="detected (unused)",
    )
    ax.scatter(
        df.loc[selected, "img_loc_x"],
        df.loc[selected, "img_loc_y"],
        s=6,
        c="tab:blue",
        label="used in calibration",
    )

    for i in range(1, COVERAGE_GRID):
        ax.axvline(width * i / COVERAGE_GRID, color="0.6", lw=0.5)
        ax.axhline(height * i / COVERAGE_GRID, color="0.6", lw=0.5)

    ax.set(
        xlim=(0, width),
        ylim=(height, 0),
        aspect="equal",
        xlabel="x (px)",
        ylabel="y (px)",
        title=(
            f"Corner coverage: {report_.coverage_fraction:.0%} "
            f"(edges {report_.edge_coverage_fraction:.0%}, "
            f"corners {report_.corner_coverage_fraction:.0%})"
        ),
    )
    ax.legend(loc="upper right", fontsize="small", markerscale=2)


def plot_per_frame_error(residuals, rmse, ax) -> None:
    """Per-frame RMSE -- a few tall bars usually mean a blurred or mis-detected frame."""
    frames = [str(sync_index) for sync_index, _ in residuals]
    errors = [float(np.sqrt((res**2).sum(axis=1).mean())) for _, res in residuals]

    ax.bar(frames, errors, color="tab:blue")
    ax.axhline(rmse, color="tab:red", ls="--", label=f"overall RMSE = {rmse:.3f} px")
    ax.set(
        xlabel="frame (sync index)",
        ylabel="RMSE (px)",
        title="Per-frame reprojection error",
    )
    ax.tick_params(axis="x", labelrotation=90, labelsize=6)
    ax.legend(fontsize="small")


def plot_residual_scatter(residuals, ax) -> None:
    """Residual cloud -- should be an isotropic blob centred on zero.

    Structure (a ring, a bias, a comet) means the distortion model has not
    absorbed everything.
    """
    res = np.vstack([r for _, r in residuals])
    ax.scatter(res[:, 0], res[:, 1], s=3, alpha=0.3, c="tab:blue")
    ax.axhline(0, color="0.6", lw=0.5)
    ax.axvline(0, color="0.6", lw=0.5)
    limit = np.percentile(np.abs(res), 99.5)
    ax.set(
        xlim=(-limit, limit),
        ylim=(-limit, limit),
        aspect="equal",
        xlabel="x residual (px)",
        ylabel="y residual (px)",
        title=f"Reprojection residuals (n = {len(res)})",
    )


def plot_distortion_field(camera, ax) -> None:
    """How far the lens moves each pixel -- the magnitude of the fitted distortion."""
    width, height = camera.size
    matrix = np.asarray(camera.matrix, dtype=np.float64)
    dist = np.asarray(camera.distortions, dtype=np.float64).ravel()

    step = 40
    xs = np.arange(0, width, step, dtype=np.float32)
    ys = np.arange(0, height, step, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(xs, ys)
    distorted = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1).reshape(-1, 1, 2)

    if camera.fisheye:
        undistorted = cv2.fisheye.undistortPoints(distorted, matrix, dist, P=matrix)
    else:
        undistorted = cv2.undistortPoints(distorted, matrix, dist, P=matrix)

    displacement = (undistorted - distorted).reshape(-1, 2)
    magnitude = np.linalg.norm(displacement, axis=1).reshape(grid_x.shape)

    mesh = ax.pcolormesh(grid_x, grid_y, magnitude, cmap="viridis", shading="auto")
    ax.quiver(
        grid_x,
        grid_y,
        displacement[:, 0].reshape(grid_x.shape),
        displacement[:, 1].reshape(grid_x.shape),
        color="w",
        scale_units="xy",
        angles="xy",
        scale=1,
        width=0.002,
    )
    ax.figure.colorbar(mesh, ax=ax, label="displacement (px)", fraction=0.03, pad=0.02)
    ax.set(
        xlim=(0, width),
        ylim=(height, 0),
        aspect="equal",
        xlabel="x (px)",
        ylabel="y (px)",
        title="Undistortion displacement field",
    )


def plot_undistorted_frame(config, camera, sync_index, axes) -> bool:
    """A calibration frame before and after undistortion -- the eyeball check."""
    capture = cv2.VideoCapture(str(config.calibration_video))
    capture.set(cv2.CAP_PROP_POS_FRAMES, int(sync_index))
    ok, frame = capture.read()
    capture.release()
    if not ok:
        return False

    matrix = np.asarray(camera.matrix, dtype=np.float64)
    dist = np.asarray(camera.distortions, dtype=np.float64).ravel()
    if camera.fisheye:
        map_x, map_y = cv2.fisheye.initUndistortRectifyMap(
            matrix, dist, np.eye(3), matrix, camera.size, cv2.CV_32FC1
        )
        undistorted = cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR)
    else:
        undistorted = cv2.undistort(frame, matrix, dist)

    for ax, image, title in zip(
        axes, [frame, undistorted], ["original", "undistorted"]
    ):
        ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        ax.set(title=f"Frame {sync_index}: {title}")
        ax.axis("off")
    return True


def assess_quality(report_) -> tuple[str, list[str], list[str]]:
    """Turn the report's numbers into a verdict and a to-do list.

    Returns (verdict, checks, actions). Each check is one PASS/FAIL line; each
    action says what to do differently when re-recording. Thresholds match
    caliscope's own (see _RMSE_THRESHOLDS and friends in caliscope.reporting).
    """
    checks, actions = [], []

    def check(ok: bool, label: str, value: str, action: str) -> None:
        checks.append(f"[{'PASS' if ok else 'FAIL'}]  {label:<22} {value}")
        if not ok:
            actions.append(action)

    check(
        report_.rmse < 1.0,
        "reprojection error",
        f"{report_.rmse:.3f} px  (want < 1.0)",
        "High reprojection error: re-record with the camera and board held "
        "steadier (motion blur softens the corners), and in brighter, even light.",
    )
    check(
        report_.coverage_fraction >= 0.80,
        "image coverage",
        f"{report_.coverage_fraction:.0%}  (want > 80%)",
        "Sparse coverage: walk the board through the whole field of view, "
        "checking the coverage panel for the empty cells.",
    )
    check(
        report_.edge_coverage_fraction >= 0.75,
        "edge coverage",
        f"{report_.edge_coverage_fraction:.0%}  (want > 75%)",
        "Weak edge coverage: take the board right out to the edges of frame, "
        "even where it is only partly visible.",
    )
    check(
        report_.corner_coverage_fraction >= 0.50,
        "corner coverage",
        f"{report_.corner_coverage_fraction:.0%}  (want > 50%)",
        "Weak corner coverage: get the board into all four image corners. "
        "The corners are where distortion is strongest, so without them the "
        "distortion model is extrapolating.",
    )
    check(
        report_.orientation_sufficient,
        "pose diversity",
        f"{report_.orientation_count}/8 tilt directions  (want >= {MIN_ORIENTATIONS})",
        "MORE POSE DIVERSITY NEEDED: the board was held too flat-on to the "
        "camera. Tilt it steeply (~45 degrees) about several different axes -- "
        "left, right, top-away, bottom-away -- and rotate it in its own plane "
        "between takes. Working closer to the camera helps: the same tilt "
        "counts for more when the board fills more of the frame.",
    )
    check(
        report_.frames_used >= 20,
        "frames used",
        f"{report_.frames_used}  (want >= 20)",
        "Few usable frames: record for longer, and keep the whole board in "
        "shot so more corners are detected per frame.",
    )

    n_failed = len(actions)
    if n_failed == 0:
        verdict = "GOOD -- usable as is"
    elif n_failed <= 2:
        verdict = "FAIR -- usable, but worth re-recording"
    else:
        verdict = "POOR -- re-record before using"

    return verdict, checks, actions


def plot_quality_summary(report_, camera, ax) -> None:
    """Verdict, checks and next steps as text, so the PNG stands alone."""
    verdict, checks, actions = assess_quality(report_)
    fx, fy = camera.matrix[0, 0], camera.matrix[1, 1]
    cx, cy = camera.matrix[0, 2], camera.matrix[1, 2]

    lines = [f"VERDICT: {verdict}", ""] + checks
    lines += [
        "",
        f"fx, fy        {fx:.1f}, {fy:.1f} px",
        f"cx, cy        {cx:.1f}, {cy:.1f} px",
        f"image size    {camera.size[0]} x {camera.size[1]} px",
        f"distortions   {np.array2string(np.asarray(camera.distortions), precision=4)}",
    ]
    if actions:
        lines += ["", "WHAT TO DO NEXT:"]
        for i, action in enumerate(actions, start=1):
            lines += _wrap(f"{i}. {action}", width=64, indent="   ")

    ax.text(0, 1, "\n".join(lines), va="top", ha="left", family="monospace", fontsize=8)
    ax.set(title="Calibration quality")
    ax.axis("off")


def _wrap(text: str, width: int, indent: str = "") -> list[str]:
    """Wrap text to `width` columns, indenting every line after the first."""
    import textwrap

    return textwrap.wrap(text, width=width, subsequent_indent=indent) or [""]


def make_plots(config: Config, cameras, points, output, plots_path: Path) -> None:
    camera, rep = cameras[0], output.report
    per_frame = _frame_arrays(points, rep.selected_frames)
    residuals = _reprojection_residuals(camera, per_frame)

    fig = plt.figure(figsize=(14, 21))
    grid = fig.add_gridspec(4, 2)
    axes = np.array([[fig.add_subplot(grid[r, c]) for c in range(2)] for r in range(4)])

    plot_corner_coverage(points, rep, camera, axes[0, 0])
    plot_distortion_field(camera, axes[0, 1])
    if residuals:
        plot_per_frame_error(residuals, rep.rmse, axes[1, 0])
        plot_residual_scatter(residuals, axes[1, 1])

    # The verdict reads as one block, so give it the full width of the row
    axes[2, 0].remove()
    axes[2, 1].remove()
    plot_quality_summary(rep, camera, fig.add_subplot(grid[2, :]))

    if not plot_undistorted_frame(config, camera, rep.selected_frames[0], axes[3]):
        for ax in axes[3]:
            ax.axis("off")

    fig.suptitle(
        f"{config.calibration_video.name}  |  RMSE {rep.rmse:.3f} px  |  "
        f"{rep.frames_used} frames  |  fisheye={camera.fisheye}"
    )
    fig.tight_layout()
    fig.savefig(plots_path, dpi=150)
    print(f"Saved diagnostic plots to {plots_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="Path to the YAML config file")
    parser.add_argument(
        "video",
        type=Path,
        help=(
            "Calibration video: a filename inside the config's calibration_dir, "
            "or a path to the file"
        ),
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip the diagnostic plots",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show the plots in a window",
    )
    args = parser.parse_args()

    config = Config.from_yaml(args.config, args.video)
    cameras, points, output = run_calibration(config)
    report(cameras, output)

    config.output_dir.mkdir(parents=True, exist_ok=True)
    stem = config.calibration_video.stem
    intrinsics_path = config.output_dir / f"{stem}_intrinsics.toml"
    cameras.to_toml(intrinsics_path)
    print(f"\nSaved intrinsics to {intrinsics_path}")

    if config.make_plots and not args.no_plots:
        make_plots(
            config,
            cameras,
            points,
            output,
            config.output_dir / f"{stem}_diagnostics.png",
        )
        if args.show:
            plt.show()


if __name__ == "__main__":
    main()
