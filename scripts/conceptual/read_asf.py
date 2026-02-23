import os
import sys

_r = os.path.dirname(os.path.abspath(__file__))
while _r and not os.path.isfile(os.path.join(_r, "amc_parser.py")):
    _r = os.path.dirname(_r)
if _r and _r not in sys.path:
    sys.path.insert(0, _r)

import amc_parser as amc

# Path relative to project root (run from project root, or use os.path.join(_r, "subjects", ...))
asf_path = os.path.join(_r, "subjects", "01", "01.asf") if _r else "subjects/01/01.asf"

joints = amc.parse_asf(asf_path)

for joint in joints.values():
    print(joint.name)
