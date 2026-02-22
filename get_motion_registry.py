### Gets the motion registry for the given subject and motion number. 
# This is used to determine which joints are involved in the motion and how they are connected.
# Ideally we have one asf file that points to a bunch of amc files, but for now we just hardcode the paths.

import os
import glob

def get_motion_registry(root_dir):
    registry = {}
    # Find all ASF files first (one per subject)
    asf_files = glob.glob(os.path.join(root_dir, "**/*.asf"), recursive=True)
    
    for asf in asf_files:
        subject_dir = os.path.dirname(asf)
        # Find all AMCs belonging to this subject's skeleton
        amc_files = glob.glob(os.path.join(subject_dir, "*.amc"))
        registry[asf] = amc_files
        
    return registry

# Usage
motion_db = get_motion_registry("subjects")
print(f"Indexed {len(motion_db)} subjects.")