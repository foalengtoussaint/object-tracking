"""drink_study spine: the load-bearing modules imported across the study.

Scripts import these by BARE name (e.g. `import segment_cup_only`); the path shim
(see _paths.py and the per-script shim) puts this dir on sys.path, so bare imports
resolve from any subdirectory. Not a conventional package \u2014 do not rely on `from lib import x`.
"""
