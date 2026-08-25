

# %%
from pathlib import Path

import cv2
import numpy as np
import stag
from caliscope.api import CameraArray

# stag.detectMarkers segfaults on some real frames (an upstream buffer overrun
# in its edge detector, unfixed as of stag-python 1.1.1). Running it in a
# subprocess keeps a bad frame from killing the kernel; see
# crab_stags/stag_safe.py.
from crab_stags.stag_safe import StagDetector

# The marker centre is the intersection of the image diagonals, undistorted
# through the calibrated lens model rather than averaged over the corners.
# See crab_stags/markers.py for why; scripts/detect_markers.py uses the same
# helper, so the notebook and the script report identical centres.
from crab_stags.markers import marker_centres

# %%%%%%%%%%%%%%%%%%%%
# Marker: input data

# Set library or family of tags to use
# https://github.com/manfredstoiber/stag-python#-configuration
# errorCorrection is in the range 0 <= errorCorrection <= (libraryHD-1)/2
stag_libraryHD = 15



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
    "/Users/sofia/swc/project_stags/crab_stags/calibration/calibration_video_20260825_1027_intrinsics.toml"
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

# whether to annotate each detected corner with its index
# useful to confirm the 2D corner order matches obj_points
check_corner_order = True


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


# Run loop

# Marker detection, isolated in a subprocess so an upstream segfault costs us
# one frame instead of the kernel. Frames lost this way are listed in
# detector.skipped and reported after the loop.
detector = StagDetector(libraryHD=stag_libraryHD)

# One row per detected marker per frame: (frame, marker_id, centre_x, centre_y),
# with the centre in pixels of the raw (still distorted) image.
detections = []

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
        # Same signature and return as stag.detectMarkers, but a frame that
        # crashes the detector comes back as an empty detection rather than
        # taking the kernel down with it.
        corners, ids, rejected = detector.detect(image)
        # (corners, ids, rejected_corners) = stag.detectMarkers(image, stag_libraryHD) -- crashes


        # Get the id and the pixel coordinates of the centre of every marker
        # detected in this frame. Passing `camera` undistorts the corners before
        # the diagonals are intersected, which is what makes the centre exact
        # rather than off by up to ~1 px near the frame edge.
        # A frame with no markers (or one the detector crashed on) gives [].
        frame_centres = marker_centres(corners, ids, camera)

        for marker_id, centre_x, centre_y in frame_centres:
            detections.append((frame_idx, marker_id, centre_x, centre_y))

            # Mark the centre. drawDetectedMarkers below prints the id midway
            # between corners 0 and 2, which is the corner mean and so sits a
            # little off this cross on a tilted marker -- that gap is exactly
            # the systematic offset marker_centre avoids.
            cv2.drawMarker(
                image,
                (int(round(centre_x)), int(round(centre_y))),
                (0, 0, 255),
                cv2.MARKER_CROSS,
                markerSize=20,
                thickness=2,
            )

        # Draw markers
        # the dot marks corner 0, the id is printed midway between
        # corners 0 and 2
        stag.drawDetectedMarkers(image, corners, ids)

        # Visual check of the corner order
        # stag's drawing does not distinguish corner 1 from corner 3,
        # so label the indices to confirm 0,1,2,3 walk around the marker
        # in the same order as obj_points
        # If the walk direction or the starting corner is off, 
        # apply the constant np.roll on img_points rather than touching obj_points — SOLVEPNP_IPPE_SQUARE pins that ordering.
        if check_corner_order:
            for marker_corners in corners:
                pts = np.asarray(marker_corners, dtype=np.float32).reshape(4, 2)
                for corner_idx, (x, y) in enumerate(pts):
                    cv2.putText(
                        image,
                        str(corner_idx),
                        (int(x) + 8, int(y) - 8),  # offset to clear the corner dot
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2,
                    )

        # Add labelled frame to video
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
    # Frames go into the file as the loop runs, so release() only has to
    # finalise the container -- it should take milliseconds, not seconds. Each
    # step is timed because the wait after pressing q has to come from one of
    # them, and they are easy to confuse with "saving the video".
    cap.release()

    if writer is not None:
        writer.release()

    if show_preview:
        cv2.destroyAllWindows()
        # macOS only really closes the window once the GUI event loop gets a
        # few more turns; without this the window lingers and the wait looks
        # like the file still being written.
        for _ in range(4):
            cv2.waitKey(1)

    detector.close()


# Frames the detector crashed on carry no detections, so they are indistinguishable
# from empty frames in the output video. Report them explicitly: a high count means
# the results are sparse for a reason that has nothing to do with the markers.
if detector.skipped:
    print(
        f"\n{len(detector.skipped)} of {frame_idx} frames were skipped after the "
        f"stag detector crashed on them (upstream bug, see crab_stags/stag_safe.py)."
    )
    print(f"skipped frame indices: {detector.skipped}")

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Save the centres alongside the annotated video
# set to None to skip
centres_path = "marker_centres.csv"

if centres_path is not None:
    np.savetxt(
        centres_path,
        np.array(detections, dtype=np.float64).reshape(-1, 4),
        delimiter=",",
        header="frame,marker_id,centre_x,centre_y",
        comments="",
        # frame and marker_id are integers, the centres are sub-pixel
        fmt=["%d", "%d", "%.4f", "%.4f"],
    )
    print(f"\nSaved {len(detections)} centres to {centres_path}")

# %%
