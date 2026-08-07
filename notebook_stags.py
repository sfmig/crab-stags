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

# %%%%%%%%%%%%%%%%%%%%
# Marker: input data

# the units used here determine the sale for tvec etc
marker_side_m = 0.05  # meters, example: 5 cm

# Set library or family of tags to use
# https://github.com/manfredstoiber/stag-python#-configuration
# errorCorrection is in the range 0 <= errorCorrection <= (libraryHD-1)/2
stag_libraryHD = 15


# 3D marker corners in the marker coordinate frame
# origin: marker centre
# order must match the detected 2D corner order from stag
# obj_points must be the cycle 0→1→2→3 going around the square, with row 0 at the marker-frame origin corner
obj_points = np.array([
    [-marker_side_m / 2, -marker_side_m / 2, 0],  # (0,0)
    [ marker_side_m / 2, -marker_side_m / 2, 0],  # (1,0)
    [ marker_side_m / 2,  marker_side_m / 2, 0],  # (1,1)
    [-marker_side_m / 2,  marker_side_m / 2, 0],  # (0,1)
], dtype=np.float32)


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Input data: camera calibration
# For solvePnP, OpenCV expects
# cameraMatrix = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]], and distCoeffs
# in the form (k1, k2, p1, p2[, k3[, k4, k5, k6[, s1, s2, s3, s4[, τx, τy]]]]).
# If distCoeffs is empty or None, OpenCV assumes zero distortion. (https://docs.opencv.org/3.3.1/d9/d0c/group__calib3d.html)
# https://docs.opencv.org/3.3.1/d9/d0c/group__calib3d.html#:~:text=In%20the%20functions%20below%20the%20coefficients%20are%20passed%20or%20returned%20as
#
# fx, fy, cx, cy are all in pixels, at the resolution you calibrated at
# dist_coeffs are dimensionless
#
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



# %%%%%%%%%%%%%%%%%%%%%%%%%%%
# Inputs for video source
# - an int is a camera index (0 = default webcam), relevant for live stream
# - also accepts a video file path (relevant for offline processing)
video_source = 0

# where to write the annotated video
# set to None to skip writing
output_path = "pose_result.mp4"

# whether to show a live preview window (press q to stop)
show_preview = True

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Initialise loop

# Open the video stream
cap = cv2.VideoCapture(video_source)
if not cap.isOpened():
    raise RuntimeError(f"Could not open video source: {video_source}")

# Set stream properties
# (needed to configure the writer)
frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
if not fps or np.isnan(fps) or fps <= 0:
    # webcams often report 0 fps
    fps = 30.0

# Initialise writer
writer = None
if output_path is not None:
    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (frame_w, frame_h),
    )

# %%%%%%%%%%%%%%%%%%%%%%%%%%%
# Run loop

# Process every frame
frame_idx = 0
try:
    while True:
        ret, image = cap.read()
        if not ret:
            # end of file, or dropped camera
            break

        # Detect markers
        # corners: A list containing the (x, y)-coordinates of our detected ArUco markers
        # ids: The ArUco IDs of the detected markers
        corners, ids, rejected = stag.detectMarkers(image, stag_libraryHD)

        # Draw markers
        # white dot is corner 0;
        # confirm corners 1,2,3 walk around the marker in the order
        # specified in obj_points
        stag.drawDetectedMarkers(image, corners, ids)

        # If at least one marker detected, loop thru them
        if ids is not None and len(ids) > 0:
            for marker_corners, marker_id in zip(corners, ids.flatten()):
                # get coordinates of markers in the image coord system
                img_points = np.asarray(marker_corners, dtype=np.float32).reshape(4, 2)

                # TODO: undistort image points if fisheye

                # solve PnP
                # https://docs.opencv.org/4.13.0/d5/d1f/calib3d_solvePnP.html
                # rvec and tvec bring points from the model coord system 
                # (i.e. 3d coord system of the marker), to the camera coord system
                success, rvec, tvec = cv2.solvePnP(
                    obj_points,
                    img_points,
                    camera_matrix,
                    dist_coeffs,
                    flags=cv2.SOLVEPNP_IPPE_SQUARE, # if square marker, required 4 points
                    # IPPE_SQUARE expects the specific corner ordering 
                    # (top-left, top-right, bottom-right, bottom-left in the marker frame)
                )

                if success:

                    # Express rotation as matrix
                    R_matrix, _ = cv2.Rodrigues(rvec)

                    # print(f"Frame {frame_idx}, marker {marker_id}")
                    # print("tvec (x, y, z) in meters:")
                    # print(tvec.flatten())
                    # print("rvec:")
                    # print(rvec.flatten())
                    # print("rotation matrix:")
                    # print(R_matrix)

                    # Draw axes of marker 3d coord system
                    cv2.drawFrameAxes(
                        image,
                        camera_matrix,
                        dist_coeffs,
                        rvec,
                        tvec,
                        marker_side_m * 0.5,
                        # Length of the painted axes in the same unit than tvec (usually in meters).
                        # thickness = 3 (default)
                    )

        # Add frame to video
        if writer is not None:
            writer.write(image)

        # Show live preview if required 
        if show_preview:
            cv2.imshow("pose", image)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        # Update         
        frame_idx += 1

# A finally block runs no matter how the "try" block ends
finally:
    cap.release()
    if writer is not None:
        writer.release()
    if show_preview:
        cv2.destroyAllWindows()
