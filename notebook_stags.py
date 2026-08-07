"""

Recommended flow:
1. Load the camera intrinsics calibrated in notebook_calibrate_intrinsic_one_camera.py
2. Define the marker's 3D corner coordinates in its local coordinate system
3. Detect markers with stag.detectMarkers(...)
4. Undistort the detected 2D corners into normalized coordinates
5. Run cv2.solvePnPGeneric(...) with identity K and zero distortion, and resolve
   the two-fold planar pose ambiguity it returns (see select_pose)
6. Convert the returned rotation vector to a rotation matrix with cv2.Rodrigues(...)

The corner order must match between your 3D obj_points and the 2D corners returned by STag.

"""

# %%
from pathlib import Path

import cv2
import numpy as np
import stag
from caliscope.api import CameraArray

# stag.detectMarkers segfaults on some real frames (an upstream buffer overrun
# in its edge detector, unfixed as of stag-python 1.1.1). Running it in a
# subprocess keeps a bad frame from killing the kernel; see stag_safe.py.
from stag_safe import StagDetector

# project_points dispatches between the Brown-Conrady and fisheye equidistant
# models. Not part of caliscope.api's public surface, but it is the same helper
# the library uses internally for all of its own reprojection.
from caliscope.core.reprojection import project_points

# %%%%%%%%%%%%%%%%%%%%
# Marker: input data

# the units used here determine the sale for tvec etc
marker_side_m = 0.10  # meters

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
    "/Users/sofia/swc/project_stags/calibration_video_20260807_1823_intrinsics.toml"
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
# Helper: resolve the planar pose ambiguity
#
# Four coplanar points constrain only the homography, and decomposing a
# homography gives two poses that satisfy the cheirality constraint. Both
# reproject the corners almost equally well, so a single-solution solver picks
# between them on noise alone and the marker's normal flips frame to frame.
#
# The two poses are related by a reflection of the marker normal about the line
# of sight: the tilt away from the camera keeps its magnitude and swaps its
# sign. What separates them is perspective foreshortening across the marker,
# of relative size
#
#     eps ~ (marker size) * sin(tilt) / (distance)
#
# For a 5 cm marker at 1.5 m that is around 1%, i.e. sub-pixel, hence the
# instability at small, distant or near-frontal markers.
#
# STag and ARToolKit+ handle this by computing both solutions and keeping the
# one with the lower error (the RPP method of Schweighofer & Pinz, "Robust pose
# estimation from a planar target", TPAMI 2006). The STag pose estimation approach
# described in the paper is not exposed by stag-python (we only get corners), so we do
# something equivalent here with cv2.solvePnPGeneric. While RPP uses object-space error
# (measured along the ray in the scene), we measure reprojection error in the image plane.
# For selecting between two candidates the two metrics almost always agree.
#
# RPP stops there: lower error wins, decided fresh each frame with no history.
# But the error test alone is least reliable in exactly the regime that causes
# flipping. So beyond RPP we additionally use the ratio of the two errors as an
# ambiguity *detector*, and fall back to temporal continuity when the frame
# cannot decide on its own.


def rotation_angle_between(rvec_a, rvec_b):
    """Geodesic angle in radians between two rotations, as Rodrigues vectors."""
    R_a, _ = cv2.Rodrigues(np.asarray(rvec_a, dtype=np.float64).reshape(3, 1))
    R_b, _ = cv2.Rodrigues(np.asarray(rvec_b, dtype=np.float64).reshape(3, 1))
    # R_a.T @ R_b is the rotation that takes you from frame a to frame b, expressed in a's frame.
    # cv2.Rodrigues on that matrix gives the axis-angle vector, whose norm is the geodesic angle
    # between the two orientations. Geodesic (angular) is the arc length of the shortest path within SO(3)
    # between the two orientations R_a and R_b
    r_rel, _ = cv2.Rodrigues(R_a.T @ R_b)
    return float(np.linalg.norm(r_rel))


def select_pose(rvecs, tvecs, errors, prev_rvec, ratio_threshold):
    """Choose between the candidate poses returned by cv2.solvePnPGeneric.

    Parameters
    ----------
    rvecs, tvecs : sequence of (3, 1) arrays
        Candidate poses.
    errors : sequence of float
        Per-candidate RMS reprojection error, in the units solvePnPGeneric was
        called with (normalized here, i.e. focal lengths, not pixels).
    prev_rvec : (3, 1) array or None
        Accepted rotation for this marker on a recent frame, if any.
    ratio_threshold : float
        Below this, errors[0] / errors[1] is treated as decisive.

    Returns
    -------
    rvec, tvec, info
        info carries err_ratio, whether the frame was ambiguous, and which rule
        made the choice ("only-solution", "error", "continuity" or
        "error-no-history").
    """
    order = np.argsort(np.asarray(errors, dtype=np.float64).ravel())
    rvecs = [rvecs[i] for i in order]
    tvecs = [tvecs[i] for i in order]
    errors = [float(np.ravel(errors[i])[0]) for i in order]

    if len(rvecs) == 1:
        return rvecs[0], tvecs[0], {
            "err_ratio": 0.0,
            "ambiguous": False,
            "chose_by": "only-solution",
        }

    # errors[0] <= errors[1] after the sort, so the ratio is in [0, 1].
    # Near 1 means the two candidates explain the corners equally well.
    # Guard the degenerate case of a numerically exact fit on both.
    err_ratio = errors[0] / errors[1] if errors[1] > 0 else 1.0
    ambiguous = err_ratio >= ratio_threshold

    if not ambiguous:
        # The perspective signal is above the noise: trust it. Deliberately
        # checked before continuity so a genuine flip of the marker is followed
        # rather than suppressed, and so a wrong lock cannot persist forever.
        return rvecs[0], tvecs[0], {
            "err_ratio": err_ratio,
            "ambiguous": False,
            "chose_by": "error",
        }

    if prev_rvec is None:
        # Ambiguous and nothing to fall back on. Take the lower error, but say so.
        return rvecs[0], tvecs[0], {
            "err_ratio": err_ratio,
            "ambiguous": True,
            "chose_by": "error-no-history",
        }

    # Ambiguous: pick the candidate closest to where this marker just was.
    # Only rotation is compared, since the two solutions differ mainly in the
    # normal and share nearly the same translation.
    angles = [rotation_angle_between(prev_rvec, rvec) for rvec in rvecs]
    best = int(np.argmin(angles))
    return rvecs[best], tvecs[best], {
        "err_ratio": err_ratio,
        "ambiguous": True,
        "chose_by": "continuity",
    }


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

# %%%%%%%%%%%%%%%%%%%%%%%%%%%
# Inputs for pose ambiguity handling

# How decisive the two reprojection errors have to be, as errors[0] / errors[1]
# with errors[0] the smaller. 0 means one candidate fits perfectly and the other
# does not; 1 means they are indistinguishable. If the ratio is below this
# threshold we trust the errors; if it is at or above, we treat the frame as
# ambiguous and fall back to continuity.
# 0.6 is the usual starting point; raise it to lean harder on
# continuity, lower it to lean harder on the current frame.
ambiguity_ratio_threshold = 0.6

# How many frames a marker may go undetected before its remembered pose is
# considered stale. Keeps continuity from being seeded by where a marker was
# several seconds ago, e.g. after an occlusion.
pose_memory_frames = 5

# whether to print a line each time a frame is ambiguous
report_ambiguity = True

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

# Last accepted pose per marker id, as {marker_id: (frame_idx, rvec)}.
# Read by select_pose to break ties that the current frame cannot.
prev_poses = {}

# Marker detection, isolated in a subprocess so an upstream segfault costs us
# one frame instead of the kernel. Frames lost this way are listed in
# detector.skipped and reported after the loop.
detector = StagDetector(libraryHD=stag_libraryHD)

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

        # A crashed frame carries no detections, making it indistinguishable in
        # the video from a frame the marker simply left. Label it, so a gap in
        # the pose track can be read as a detector failure rather than as the
        # marker moving out of view. detector.skipped holds frame indices (one
        # detect call per frame, both counters starting at 0), and is appended
        # to in order, so the last entry is this frame iff it just crashed.
        frame_was_skipped = bool(detector.skipped) and detector.skipped[-1] == frame_idx
        if frame_was_skipped:
            cv2.putText(
                image,
                "DETECTOR CRASHED",
                (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                (0, 0, 255),
                3,
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
                # solvePnPGeneric returns every solution the flag admits, rather
                # than the single one solvePnP picks. For IPPE_SQUARE that is the
                # two poses of the planar ambiguity, plus their reprojection
                # errors, so we can judge how separable they are (see select_pose).
                n_solutions, rvecs, tvecs, errors = cv2.solvePnPGeneric(
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

                if n_solutions > 0:
                    # Only reuse the remembered pose if it is recent: a marker
                    # that has been out of view has had time to move, and a stale
                    # pose would bias the tie-break toward a stale answer.
                    prev_entry = prev_poses.get(marker_id)
                    prev_rvec = None
                    if prev_entry is not None:
                        prev_frame_idx, remembered_rvec = prev_entry
                        if frame_idx - prev_frame_idx <= pose_memory_frames:
                            prev_rvec = remembered_rvec

                    rvec, tvec, pose_info = select_pose(
                        rvecs,
                        tvecs,
                        errors,
                        prev_rvec,
                        ambiguity_ratio_threshold,
                    )
                    prev_poses[marker_id] = (frame_idx, rvec)

                    if report_ambiguity and pose_info["ambiguous"]:
                        print(
                            f"Frame {frame_idx}, marker {marker_id}: ambiguous pose "
                            f"(error ratio {pose_info['err_ratio']:.2f}), "
                            f"resolved by {pose_info['chose_by']}"
                        )

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
    detector.close()

# Frames the detector crashed on carry no detections, so they are indistinguishable
# from empty frames in the output video. Report them explicitly: a high count means
# the results are sparse for a reason that has nothing to do with the markers.
if detector.skipped:
    print(
        f"\n{len(detector.skipped)} of {frame_idx} frames were skipped after the "
        f"stag detector crashed on them (upstream bug, see stag_safe.py)."
    )
    print(f"skipped frame indices: {detector.skipped}")

# %%
