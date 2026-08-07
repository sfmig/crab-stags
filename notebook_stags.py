"""

Recommended flow:
1. Load the camera intrinsics calibrated in notebook_calibrate_intrinsic_one_camera.py
2. Define the marker's 3D corner coordinates in its local coordinate system
3. Detect markers with stag.detectMarkers(...)
4. Undistort the detected 2D corners into normalized coordinates
5. Run cv2.solvePnP(...) with identity K and zero distortion
6. Convert the returned rotation vector to a rotation matrix with cv2.Rodrigues(...)

The corner order must match between your 3D obj_points and the 2D corners returned by STag.

"""

# %%
from pathlib import Path

import cv2
import numpy as np
import stag
from caliscope.api import CameraArray

# project_points dispatches between the Brown-Conrady and fisheye equidistant
# models. Not part of caliscope.api's public surface, but it is the same helper
# the library uses internally for all of its own reprojection.
from caliscope.core.reprojection import project_points

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
# order must match the detected 2D corner order from STag
#
# ATT! SOLVEPNP_IPPE_SQUARE assumes these 4 points are an axis-aligned planar
# square given in the order
#   (-s/2, +s/2), (+s/2, +s/2), (+s/2, -s/2), (-s/2, -s/2)
# i.e. top-left, top-right, bottom-right, bottom-left with the origin at 
# the centre of the marker and y pointing up.
# +x to the marker's right, +y to its top, +z out of the printed face toward the viewer.
# So keep this exact order as long as IPPE_SQUARE is used.
# See https://docs.opencv.org/4.13.0/d5/d1f/calib3d_solvePnP.html
obj_points = np.array(
    [
        [-marker_side_m / 2, marker_side_m / 2, 0],  # top-left
        [marker_side_m / 2, marker_side_m / 2, 0],  # top-right
        [marker_side_m / 2, -marker_side_m / 2, 0],  # bottom-right
        [-marker_side_m / 2, -marker_side_m / 2, 0],  # bottom-left
    ],
    dtype=np.float32,
)


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Input data: camera calibration
# Read the intrinsics written by notebook_calibrate_intrinsic_one_camera.py.
#
# camera.matrix = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
# fx, fy, cx, cy are all in pixels, at the resolution you calibrated at
# camera.distortions are dimensionless: (k1, k2, p1, p2, k3) for a standard lens,
# (k1, k2, k3, k4) for fisheye
# camera.fisheye says which of the two models the coefficients belong to
intrinsics_path = Path(
    "/Users/sofia/swc/project_caliscope/P2/calibration/intrinsic/cam_2_intrinsics.toml"
)
cam_id = 0

cameras = CameraArray.from_toml(intrinsics_path)
camera = cameras[cam_id]

print(f"Loaded intrinsics for camera {cam_id} from {intrinsics_path}")
print(f"fisheye: {camera.fisheye}")
print(f"calibrated at (w, h): {camera.size}")
print(camera.matrix)
print(camera.distortions)
print(f"calibration RMSE (px): {camera.error}")

# caliscope applies rotation_count 90-degree rotations to frames before
# calibrating, so a nonzero value means these intrinsics describe a rotated
# frame and would not match the raw frames read below.
if camera.rotation_count:
    raise ValueError(
        f"Camera {cam_id} was calibrated with rotation_count="
        f"{camera.rotation_count}; this notebook assumes unrotated frames."
    )

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Helper: draw the marker axes
# cv2.drawFrameAxes projects through cv2.projectPoints, i.e. always the
# Brown-Conrady model, so it silently misreads a 4-vector of fisheye
# coefficients. We project the axis endpoints ourselves instead, dispatching on
# the lens model the same way caliscope does.
_AXIS_COLORS_BGR = ((0, 0, 255), (0, 255, 0), (255, 0, 0))  # x red, y green, z blue


def draw_frame_axes(image, camera, rvec, tvec, length, thickness=3):
    """Draw the marker coordinate system onto the (distorted) frame."""
    axis_points = np.array(
        [
            [0.0, 0.0, 0.0],  # origin
            [length, 0.0, 0.0],  # x
            [0.0, length, 0.0],  # y
            [0.0, 0.0, length],  # z
        ],
        dtype=np.float64,
    )

    projected = project_points(
        axis_points,
        rvec,
        tvec,
        camera.matrix,
        camera.distortions,
        camera.fisheye,
    )

    origin = tuple(np.round(projected[0]).astype(int))
    for tip, color in zip(projected[1:], _AXIS_COLORS_BGR):
        cv2.line(image, origin, tuple(np.round(tip).astype(int)), color, thickness)


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

# Check the resolution of the stream matches the one used in the camera calibration.
# fx, fy, cx, cy are in pixels at the calibration resolution, so intrinsics
# from a different resolution are silently wrong rather than obviously broken.
if (frame_w, frame_h) != tuple(camera.size):
    cap.release()
    raise ValueError(
        f"Video source is {frame_w}x{frame_h} but the intrinsics were "
        f"calibrated at {camera.size[0]}x{camera.size[1]}."
    )

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
                # Get coordinates of markers in the image coord system
                img_points = np.asarray(marker_corners, dtype=np.float32).reshape(4, 2)

                # Remove lens distortion before running solvePnP.
                # solvePnP has no fisheye mode: it would read our 4 fisheye
                # coefficients (k1, k2, k3, k4) from the equidistant model
                # as Brown-Conrady (k1, k2, p1, p2). To fix this, we use undistort_points.
                #
                # undistort_points returns normalized coordinates, i.e. image-plane coordinates
                # expressed after removing the effect of the intrinsic matrix K. x = K⁻¹ p, with
                # p = (u, v, 1)ᵀ being pixel point in homogenous form.
                # (K removed, principal point at the top left, units of focal lengths).
                # Equivalently, for a 3D point (X, Y, Z) in the camera frame, the normalised coordinates
                # are just the perspective projection with no intrinsics applied
                img_points_norm = camera.undistort_points(
                    img_points,
                    output="normalized",
                )

                # Solve PnP
                # https://docs.opencv.org/4.13.0/d5/d1f/calib3d_solvePnP.html
                # rvec and tvec bring points from the model coord system
                # (i.e. 3d coord system of the marker), to the camera coord system
                #
                # The points are already undistorted and normalized, so the
                # camera model K here is the identity and the distortion is zero.
                # Any other K or nonzero distortion would apply the correction a second time.
                # rvec/tvec are unaffected by the change of image coordinates and
                # stay in the units of obj_points (metres).
                success, rvec, tvec = cv2.solvePnP(
                    obj_points,
                    img_points_norm,
                    np.identity(3),
                    np.zeros(5),
                    flags=cv2.SOLVEPNP_IPPE_SQUARE,  # square marker; exactly 4 points
                    # OpenCV raises for any count other than 4 under this flag.
                    # Not a constraint in practice: STag returns either no entry for
                    # a marker or a full set of 4 corners, never a partial one (note
                    # it is quite robust to occlusions though)
                    #
                    # IPPE_SQUARE expects the specific corner ordering
                    # (top-left, top-right, bottom-right, bottom-left in the marker frame)
                )

                if success:
                    # Express rotation as matrix
                    R_matrix, _ = cv2.Rodrigues(rvec)

                    # Sanity check: project obj_points (corners in 3d marker coord system)
                    # through the full lens model and compare against the detected corners 
                    # in the image coord system.
                    # With 4 points this is not a quality metric (an exactly
                    # determined fit), but it does catch a corner-order or
                    # coordinate-frame mismatch, which shows up as hundreds of
                    # pixels rather than a fraction of one.
                    reprojected = project_points(
                        obj_points.astype(np.float64),
                        rvec,
                        tvec,
                        camera.matrix,
                        camera.distortions,
                        camera.fisheye,
                    )
                    rmse_px = float(
                        np.sqrt(((reprojected - img_points) ** 2).sum(axis=1).mean())
                    )
                    if rmse_px > 5.0:
                        print(
                            f"Frame {frame_idx}, marker {marker_id}: "
                            f"reprojection RMSE {rmse_px:.1f} px -- check that the "
                            f"stag corner order matches obj_points"
                        )


                    # Draw axes of marker 3d coord system
                    # (drawn back onto the raw, still-distorted frame, so the
                    # projection uses the full lens model rather than the
                    # identity one the pose was solved in)
                    draw_frame_axes(
                        image,
                        camera,
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
