# crab-stags

Prototype code for tracking [STag](https://github.com/bbenligiray/stag) fiducial markers on crabs: capturing calibration videos, computing camera intrinsics with [caliscope](https://github.com/mprib/caliscope), and detecting marker centres in video or live from FLIR Blackfly cameras.

This is an exploratory prototype developed alongside Claude, not a packaged tool. Nothing is stable or tested for reuse, but it could be useful for inspiration. 

The scripts in [scripts/](scripts/) are driven by the YAML files in [configs/](configs/), paths and parameters are project-specific. See [pyproject.toml](pyproject.toml) for how to get a working environment.
