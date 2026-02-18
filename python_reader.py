import numpy as np

import pathlib
import sys

from cmureader import CMUReader

data_root = "subjects"

for folder in pathlib.Path(data_root).iterdir():
    if folder.is_dir():
        asf_file = list(folder.glob("*.asf"))[0]
        reader = CMUReader(asf_file)

        for amc_file in folder.glob("*.amc"):
            motion_data = reader.load_amc(amc_file)
            # This is going to be of shape [Frames, Joints, 3]
            
