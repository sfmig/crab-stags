"""
Reference: https://github.com/mprib/caliscope/blob/main/scripts/demo_api.py

"""
# %%
from pathlib import Path

from caliscope.api import (
    CameraArray,
    Charuco,
    CharucoTracker,
    calibrate_intrinsics,
    extract_image_points,
)

# %%
# Params
calibration_video_path = Path(
    "/Users/sofia/swc/project_stags/calibration/calibration_video_20260807_1823.mp4"
)

charuco_n_cols = 4
charuco_n_rows = 5
charuco_square_sz_cm = 5.00  

# charuco_square_sz_cm = 5.40 does not propagate into the intrinsics 
# You could set it to 1.0 and get identical matrix/distortions.

# From the docs:
# Intrinsic calibration does not use physical size. 
# You can measure your target after intrinsic calibration, 
# or use different-sized boards for the two stages.

# fisheye camera
is_camera_fisheye = False # set False for normal lenses

# for 2d landmark extraction
frame_step = 1

# %%
# Create charuco tracker
charuco = Charuco.from_squares(
    columns=charuco_n_cols,
    rows=charuco_n_rows,
    square_size_cm=charuco_square_sz_cm,
) 
tracker = CharucoTracker(charuco)

# Initialise camera
# (gets pixel resolution from video)
cameras = CameraArray.from_video_metadata({0: calibration_video_path})

# set Fisheye lens
# fisheye model uses 4 distortion coefficients (k1, k2, k3, k4), assumes equidistant model.
# standard uses 5; 
# See: https://github.com/mprib/caliscope/blob/ddda95b44ba281c9bf968d2d0acbf7b0ab167e7d/src/caliscope/core/reprojection.py#L21
# ATT! cv2.solvePnP interprets a length-4 distCoeffs as (k1, k2, p1, p2) 
# in the plumb-bob/Brown-Conrady/standard model. So we need to pass undistorted points.
# (solvePnP has no fisheye mode) 
cameras[0].fisheye = is_camera_fisheye   

# Extract 2d landmarks from calibration video
points = extract_image_points(
    calibration_video_path,
    cam_id=0,
    tracker=tracker,
    frame_step=frame_step,
)

# Calibrate single camera
# NOTE: at max, the best 30 frames are used 
# (see grid_count = 30 in .toml file)
output = calibrate_intrinsics(points, cameras[0])

# overwrite camera with calibration output
cameras[0] = output.camera

# %%
# Write calibration parameters to file
print(cameras[0].matrix)
print(cameras[0].distortions)
print(cameras[0].fisheye)
print(f"RMSE (px): {output.camera.error}")

output_path = calibration_video_path.parent / f"{calibration_video_path.stem}_intrinsics.toml"
cameras.to_toml(output_path)
print(f"Saved intrinsics to {output_path}")

# %%
# TODO: visualise results?