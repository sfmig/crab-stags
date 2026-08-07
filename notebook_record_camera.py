"""
Record a video from a camera connected to this machine.

Use this to collect a calibration video (e.g. moving a charuco board around
the field of view), then feed the resulting .mp4 to
notebook_calibrate_intrinsic_one_camera.py

macOS note: the first run will ask for camera permission for the app running
this script (VSCode / Terminal). If no window appears, check
System Settings > Privacy & Security > Camera.
"""

# %%
import time
from datetime import datetime
from pathlib import Path

import cv2

# %%
# Params
camera_index = 0  # 0 is usually the built-in webcam
timestamp = datetime.now().strftime("%Y%m%d_%H%M")
output_path = Path(f"/Users/sofia/swc/project_stags/calibration_video_{timestamp}.mp4")

# Optional -- set to None to keep whatever the camera defaults to
frame_width = None
frame_height = None
fps = None

fallback_fps= 30.0  # fallback if the camera reports 0

# %%
# Check which camera indices are available
# (a bit slow -- each failed index takes a moment to time out)
for idx in range(4):
    cap = cv2.VideoCapture(idx, cv2.CAP_AVFOUNDATION)
    if cap.isOpened():
        ok, frame = cap.read()
        if ok:
            print(f"camera {idx}: available, frame shape {frame.shape}")
    else:
        print(f"camera {idx}: not available")
    cap.release()

# %%
# Open the camera and apply the requested settings
cap = cv2.VideoCapture(camera_index, cv2.CAP_AVFOUNDATION)
if not cap.isOpened():
    raise RuntimeError(f"Could not open camera {camera_index}")

if frame_width is not None:
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
if frame_height is not None:
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
if fps is not None:
    cap.set(cv2.CAP_PROP_FPS, fps)

# The camera may not honour the request -- use what it actually gives us
actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
actual_fps = cap.get(cv2.CAP_PROP_FPS) or fallback_fps
print(f"Recording at {actual_width}x{actual_height} @ {actual_fps:.1f} fps")

# %%
# Record -- press 'q' in the preview window to stop
writer = cv2.VideoWriter(
    str(output_path),
    cv2.VideoWriter_fourcc(*"mp4v"),
    actual_fps,
    (actual_width, actual_height),
)

n_frames = 0
t_start = time.time()
try:
    while True:
        ok, frame = cap.read()
        if not ok:
            print("Dropped frame from camera, stopping")
            break

        writer.write(frame)
        n_frames += 1

        cv2.imshow("recording (press q to stop)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
finally:
    writer.release()
    cv2.destroyAllWindows()
    for _ in range(4):  # nudge macOS into actually closing the window
        cv2.waitKey(1)

duration = time.time() - t_start
print(f"Saved {n_frames} frames ({duration:.1f} s) to {output_path}")
print(f"Measured fps: {n_frames / duration:.1f} (written to file as {actual_fps:.1f})")

# %%
# Release the camera when done
cap.release()

# %%
# Sanity check: read the saved video back
cap_check = cv2.VideoCapture(str(output_path))
print(f"frames: {int(cap_check.get(cv2.CAP_PROP_FRAME_COUNT))}")
print(f"fps: {cap_check.get(cv2.CAP_PROP_FPS)}")
print(
    f"size: {int(cap_check.get(cv2.CAP_PROP_FRAME_WIDTH))}"
    f"x{int(cap_check.get(cv2.CAP_PROP_FRAME_HEIGHT))}"
)
cap_check.release()

# %%
