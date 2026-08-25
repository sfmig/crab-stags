"""Geometry helpers for detected STag markers.

Detection itself lives in crab_stags/stag_safe.py; this module only turns the
four corners STag gives back into the quantities we actually track.
"""

import numpy as np

# project_points dispatches between the Brown-Conrady and fisheye equidistant
# models. Not part of caliscope.api's public surface, but it is the same helper
# the library uses internally for all of its own reprojection.
from caliscope.core.reprojection import project_points

__all__ = ["intersect_diagonals", "marker_centre", "marker_centres"]


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

    Returns None for a degenerate detection whose diagonals do not meet.
    """
    corners = np.asarray(marker_corners, dtype=np.float64).reshape(4, 2)

    if camera is None:
        centre = intersect_diagonals(corners)
        return None if centre is None else (float(centre[0]), float(centre[1]))

    # Normalized coordinates are what the camera would have seen with no lens
    # distortion and no intrinsics applied, so the diagonals are straight there.
    centre_norm = intersect_diagonals(
        np.asarray(
            camera.undistort_points(corners, output="normalized"),
            dtype=np.float64,
        ).reshape(4, 2)
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


def marker_centres(corners, ids, camera=None) -> list[tuple[int, float, float]]:
    """Pair one whole frame's detections up as (marker_id, centre_x, centre_y).

    `corners` and `ids` are exactly what StagDetector.detect returns, including
    the empty detection a frame with no markers (or a crashed one) comes back
    as. Degenerate detections without a centre are dropped.
    """
    if ids is None or len(ids) == 0:
        return []

    out = []
    for marker_corners, marker_id in zip(corners, np.asarray(ids).flatten()):
        centre = marker_centre(marker_corners, camera)
        if centre is None:
            continue
        out.append((int(marker_id), centre[0], centre[1]))
    return out
