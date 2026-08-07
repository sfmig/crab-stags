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
charuco_square_sz_cm = 5.40  # default 3.0?


# Claude says:
# charuco_square_sz_cm = 5.40 does not propagate into the intrinsics 
# — it only scales the object points, so it affects the discarded rvecs/tvecs. 
# You could set it to 1.0 and get identical matrix/distortions.

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
# fisheye model uses 4 distortion coefficients (k1, k2, k3, k4);
# standard uses 5
# ATT! cv2.solvePnP interprets a length-4 distCoeffs as (k1, k2, p1, p2) 
# in the plumb-bob/Brown-Conrady model 
cameras[0].fisheye = True   # set False for normal lenses

# Extract 2d landmarks from calibration video
points = extract_image_points(
    video_path,
    cam_id=0,
    tracker=tracker,
    frame_step=30,
)

# Calibrate single camera
output = calibrate_intrinsics(points, cameras[0])

# %%
print(output.camera.matrix)
print(output.camera.distortions)
print(f"RMSE (px): {output.camera.error}")
# output.report
# %%
