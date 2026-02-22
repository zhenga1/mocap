import pinocchio as pin
import numpy as np
import pink
from pink import Tasks
from pink.visualization import MeshcatVisualizer

# 1. Load Berkeley Humanoid URDF
urdf_path = "C:\\Users\\aaron\\IsaacSim_4.0.0\\Berkeley-Humanoid-Lite\\source\\berkeley_humanoid_lite_assets\\data\\robots\\berkeley_humanoid\\berkeley_humanoid_lite\\urdf\\berkeley_humanoid_lite.urdf"
robot = pin.RobotWrapper.BuildFromURDF(urdf_path)

# 2. Define Markers and their locations to where we want to track them
# left_foot_link =
