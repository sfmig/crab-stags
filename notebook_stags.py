"""

Recommended flow:
1. Detect markers with stag.detectMarkers(...)
2. Define the marker's 3D corner coordinates in its local coordinate system
3. Provide your camera intrinsics: camera_matrix and dist_coeffs
4. Run cv2.solvePnP(...)
5. Convert the returned rotation vector to a rotation matrix with cv2.Rodrigues(...)

The corner order must match between your 3D obj_points and the 2D corners returned by STag.
If pose looks mirrored or unstable, this is the first thing to check.

For square planar markers, cv2.SOLVEPNP_IPPE_SQUARE is usually a good flag to try.
"""

# %%
import cv2
import numpy as np
import stag

# %%
# Inputs

# the units used here determine the sale for tvec etc
marker_side_m = 0.05  # meters, example: 5 cm

# camera calibration results
# For solvePnP, OpenCV expects 
# cameraMatrix = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]], and distCoeffs 
# in the form (k1, k2, p1, p2[, k3[, k4, k5, k6[, s1, s2, s3, s4[, τx, τy]]]]). 
# If distCoeffs is empty or None, OpenCV assumes zero distortion. (https://docs.opencv.org/3.3.1/d9/d0c/group__calib3d.html)
# https://docs.opencv.org/3.3.1/d9/d0c/group__calib3d.html#:~:text=In%20the%20functions%20below%20the%20coefficients%20are%20passed%20or%20returned%20as
camera_matrix = np.array(
    [
        [800.0, 0.0, 320.0],
        [0.0, 800.0, 240.0],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float32,
)

# dist_coeffs = np.array([k1, k2, p1, p2, k3], dtype=np.float32)
dist_coeffs = np.zeros((5, 1), dtype=np.float32)

# 3D marker corners in the marker coordinate frame
# order must match the detected 2D corner order from stag
obj_points = np.array(
    [
        [-marker_side_m / 2, marker_side_m / 2, 0],
        [marker_side_m / 2, marker_side_m / 2, 0],
        [marker_side_m / 2, -marker_side_m / 2, 0],
        [-marker_side_m / 2, -marker_side_m / 2, 0],
    ],
    dtype=np.float32,
)

# %%
# Read image
# TODO: replace with video stream
# (long term: process offline?)
image = cv2.imread("example.jpg")

# Detect markers
# corners: A list containing the (x, y)-coordinates of our detected ArUco markers
# ids: The ArUco IDs of the detected markers
corners, ids, rejected = stag.detectMarkers(image, 21)

# If at least one marker detected, loop thru them
if ids is not None and len(ids) > 0:
    for marker_corners, marker_id in zip(corners, ids.flatten()):
        # get coordinates of markers in the image
        img_points = np.asarray(marker_corners, dtype=np.float32).reshape(4, 2)

        # solve PnP
        # https://docs.opencv.org/4.13.0/d5/d1f/calib3d_solvePnP.html
        success, rvec, tvec = cv2.solvePnP(
            obj_points,
            img_points,
            camera_matrix,
            dist_coeffs,
            # flags=cv2.SOLVEPNP_IPPE_SQUARE, # if square marker, required 4 points
        )

        if success:

            # express rotation as matrix
            R_matrix, _ = cv2.Rodrigues(rvec)

            print(f"Marker {marker_id}")
            print("tvec (x, y, z) in meters:")
            print(tvec.flatten())
            print("rvec:")
            print(rvec.flatten())
            print("rotation matrix:")
            print(R_matrix)

            # optional: draw axes
            cv2.drawFrameAxes(
                image, camera_matrix, dist_coeffs, rvec, tvec, marker_side_m * 0.5
            )

# save image
cv2.imwrite("pose_result.jpg", image)
