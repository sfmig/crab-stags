"""
Reference: https://github.com/mprib/caliscope/blob/main/scripts/demo_api.py

If we do this via the GUI, we need to have at least 3 videos in intrinsics and extrinsics I think
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
video_path = Path(
    "/Users/sofia/swc/project_caliscope/P2/calibration/intrinsic/cam_2.mp4"
)

charuco_n_cols = 4
charuco_n_rows = 5
charuco_square_sz_cm = 5.00  

# Claude says:
# charuco_square_sz_cm = 5.40 does not propagate into the intrinsics 
# — it only scales the object points, so it affects the discarded rvecs/tvecs. 
# You could set it to 1.0 and get identical matrix/distortions.

# From the docs:
# Intrinsic calibration does not use physical size. 
# You can measure your target after intrinsic calibration, 
# or use different-sized boards for the two stages.

# fisheye camera
is_camera_fisheye = True # set False for normal lenses

# 2d landmark extraction
frame_step = 5

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
cameras = CameraArray.from_video_metadata({0: video_path})

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
    video_path,
    cam_id=0,
    tracker=tracker,
    frame_step=frame_step,
)

# Calibrate single camera
output = calibrate_intrinsics(points, cameras[0])

# %%
# Write calibration parameters to file
print(output.camera.matrix)
print(output.camera.distortions)
print(f"RMSE (px): {output.camera.error}")

cameras[0] = output.camera
output_path = video_path.parent / f"{video_path.stem}_intrinsics.toml"
cameras.to_toml(output_path)
print(f"Saved intrinsics to {output_path}")

# %%
