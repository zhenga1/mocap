## Test to read the URDF file. Looks like its ok
import pinocchio as pin
import pink
import os
from pink.tasks import FrameTask

# 1. Load the robot
urdf_path = "C:\\Users\\aaron\\IsaacSim_4.0.0\\Berkeley-Humanoid-Lite\\source\\berkeley_humanoid_lite_assets\\data\\robots\\berkeley_humanoid\\berkeley_humanoid_lite\\urdf\\berkeley_humanoid_lite.urdf"
urdf_dir = os.path.dirname(os.path.abspath(urdf_path))
robot = pin.RobotWrapper.BuildFromURDF(urdf_path, 
                                       package_dirs=[urdf_dir, os.path.dirname(urdf_dir)])
configuration = pink.Configuration(robot.model, robot.data, robot.q0)

# 2. Define the tasks individually
# We set 'position' gain high for tracking, and a small 'orientation' gain 
# to keep the feet pointing forward.
pelvis_task = FrameTask("base", position_cost=1.0, orientation_cost=0.1)
left_foot_task = FrameTask("leg_left_ankle_roll", position_cost=1.0, orientation_cost=0.1)
right_foot_task = FrameTask("leg_right_ankle_roll", position_cost=1.0, orientation_cost=0.1)

tasks = [pelvis_task, left_foot_task, right_foot_task]
