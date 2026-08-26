# crab-stags

## numpy / scipy: `module 'numpy' has no attribute 'long'` (resolved 2026-08-26)

Not a PySpin-vs-caliscope conflict -- that premise was wrong. The culprit is scipy.

- `np.long` was removed in numpy 1.24 and *reintroduced* in numpy 2.0, so this
  error is a symptom of running numpy 1.x, not of numpy 2.
- scipy 1.18 raised its floor to `numpy>=2.0` and calls `np.long` unguarded at
  import time, in `scipy/sparse/_sputils.py:17`. With numpy pinned to 1.26 that
  import dies and takes `caliscope.api` -- and so both detect scripts -- with it:
  `crab_stags.markers` -> `caliscope.api` -> `calibrate_extrinsics` ->
  `bundle_parameterization` -> `scipy.sparse`.
- Fix: `pip install "scipy<1.18"`. 1.17.1 declares `numpy>=1.26.4,<2.7`.
  Installed into `stags-env`. caliscope only asks for `scipy>=1.10.1`, so nothing
  else was holding scipy back -- it has to be pinned explicitly.
- Verified after: 40 frames off 23025370 through `detect_markers_flir.py` under
  numpy 1.26.4, 13.5 fps detection, 0 detector crashes, camera state untouched.

### Does PySpin need numpy < 2?

Not for Spinnaker 4.4. The installation guide (see Refs) says *"NumPy version <2
is required for Python 3.10 or later"*, but that page is undated, names no SDK
version, and is stale for the wheel installed here:

- `spinnaker-python 4.4.0.246`, from `SpinnakerSDK_FULL_4.4.0.246_x64.exe`,
  declares `Requires-Dist: numpy>=2.0` -- the opposite bound.
- Tested both ways on this machine: frames grabbed from 23025370 under numpy
  2.5.2 and under numpy 1.26.4, no import warning, no ABI error, `GetNDArray`
  returning `(1080, 1440) uint8` either way.
- That is the expected behaviour for an extension built against numpy 2 headers:
  NumPy 2 keeps those runtime-compatible down to numpy 1.19. So the `>=2.0` is a
  build requirement leaking into runtime metadata rather than a hard floor.

So neither numpy direction is forced. caliscope only asks for `numpy>=1.24`.

### Which numpy to pin

- **numpy 2.x** -- no extra pins needed, and matches what spinnaker-python,
  opencv-python 4.14 and current scipy all declare. Lowest friction on upgrades.
- **numpy 1.26** -- needs `"numpy>=1.26,<2"` *and* `"scipy<1.18"`. Three
  packages' metadata disowns that combination, so an unpinned `pip install -U`
  can re-break it, possibly less legibly than this did.

Working set in `stags-env` as of 2026-08-26: numpy 1.26.4, scipy 1.17.1,
opencv-python 4.14.0.94, pandas 3.0.5, caliscope 0.11.5, stag-python 1.1.1,
EasyPySpin 2.0.1, spinnaker-python 4.4.0.246.

## Other findings

- `stag_safe.py` bounded its wait with `select()` on a pipe, which is
  POSIX-only; on Windows that raised `WinError 10093`, which `detect()` read as
  a dead worker, so *every* frame was skipped and `detect_markers.py` silently
  returned zero detections. Fixed with a reader thread on Windows.

## TODO

- review dependencies, back to requirements? in pyproject.toml they are off
- real end-to-end detection is still unverified: no physical STag has been in
  front of the camera yet, only a stubbed detector

## Refs

- PySpin installation guide: https://www.teledynevisionsolutions.com/en-gb/support/support-center/technical-guidance/iis/installing-pyspin-for-the-spinnaker-sdk/ --- signed in with Kosta's account
- https://github.com/elerac/EasyPySpin
- pillow: https://pillow.readthedocs.io/en/stable/installation/python-support.html#python-support
- pyspin gist (unused) https://gist.github.com/unkn-one/5422fd3b4daff9bd5c2d9b2d1c859e10
