# crab-stags


- TODO: PySpin needs numpy < 2 but caliscope need 2 ---> conflict for detect markers
    CLAUDE says:" Fix applied: installed scipy 1.17.1 into stags-env (declares numpy>=1.26.4,<2.7; 1.18 was the release that jumped the floor). Verified after: 40 frames off camera 23025370 under numpy 1.26.4, 13.5 fps, 0 detector crashes, camera state untouched." -- to check
     When you re-enable it, the numpy<2 line-up needs "numpy>=1.26,<2" plus an explicit "scipy<1.18" — caliscope only asks for scipy>=1.10.1
- TODO: review dependencies, back to requirements? in pyproject.toml they are off -- read PySpin installation guide for caveats
Refs
- PySpin installataion guide: https://www.teledynevisionsolutions.com/en-gb/support/support-center/technical-guidance/iis/installing-pyspin-for-the-spinnaker-sdk/ --- signed in with Kosta's account
- https://github.com/elerac/EasyPySpin
- pillow: https://pillow.readthedocs.io/en/stable/installation/python-support.html#python-support
- pyspin gist (unused) https://gist.github.com/unkn-one/5422fd3b4daff9bd5c2d9b2d1c859e10