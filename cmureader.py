import numpy as np
import os

# The conceptual class structure for reading the CMU dataset.
class CMUReader:
    def __init__(self, asf_path):
        self.skeleton = self.load_asf(asf_path)
    
    def parse_asf(self, path):
        # Read the bone length and hierarchy
        # allows us to know the "tibia" is connected to "femur"
        pass
    def load_amc(self, amc_path):
        # Extract frame by frame joint rotations
        # Returns numpy array of shape [Frames, Joints, 3]
        pass
    
    def get_cartesian_coordinates(self, amc_frames):
        # This is 'Forward Kinematics' step
        # Convert local rotations to 3D (x,y,z) points
        # What you need to do for the IK solver
        pass


