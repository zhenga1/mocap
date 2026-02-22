import amc_parser as amc
import pinocchio as pin
import numpy as np
import pink
import os
from pink.tasks import FrameTask
## SETUP loop
amc_path = '../subjects/01/01_05.amc'
asf_path = '../subjects/01/01.asf'
urdf_path = "C:\\Users\\aaron\\IsaacSim_4.0.0\\Berkeley-Humanoid-Lite\\source\\berkeley_humanoid_lite_assets\\data\\robots\\berkeley_humanoid\\berkeley_humanoid_lite\\urdf\\berkeley_humanoid_lite.urdf"

motions = amc.parse_amc(amc_path)
joints = amc.parse_asf(asf_path)
# import pdb; pdb.set_trace()
urdf_dir = os.path.dirname(os.path.abspath(urdf_path))

## Define robot and tasks
robot = pin.RobotWrapper.BuildFromURDF(urdf_path, package_dirs=[urdf_dir, os.path.dirname(urdf_dir)])
pelvis_task = FrameTask("base", position_cost=1.0, orientation_cost=0.1)
left_foot_task = FrameTask("leg_left_ankle_roll", position_cost=1.0, orientation_cost=0.1)
right_foot_task = FrameTask("leg_right_ankle_roll", position_cost=1.0, orientation_cost=0.1)
configuration = pink.Configuration(robot.model, robot.data, robot.q0)

retargeted_data = []

for frame in motions:
    # A. Update the Human Skeleton
    # This calculates where the human's feet are in THIS frame
    joints['root'].set_motion(frame)
    
    # B. Extract Human Positions (with Axis Swap & Scale)
    human_lfoot = joints['lfoot'].coordinate
    human_rfoot = joints['rfoot'].coordinate
    human_pelvis = joints['root'].coordinate # pelvis is the root joint
    # We map Human Y -> Robot Z (Up)
    target_pos = np.array([human_lfoot[0], human_lfoot[2], human_lfoot[1]]) * 0.45
    target_rfoot = np.array([human_rfoot[0], human_rfoot[2], human_rfoot[1]]) * 0.45
    target_pelvis = np.array([human_pelvis[0], human_pelvis[2], human_pelvis[1]]) * 0.45

    # C. Tell the Robot's Link to go to that Human Position
    # This is the "Linkup"
    left_foot_task.set_target(pin.SE3(np.eye(3), target_pos))
    right_foot_task.set_target(pin.SE3(np.eye(3), target_rfoot))
    pelvis_task.set_target(pin.SE3(np.eye(3), target_pelvis))

    ## Define the right tasks to solve for
    tasks_to_solve = [pelvis_task, left_foot_task, right_foot_task]
    
    # D. Solve the Math
    # Pink calculates the exact joint angles (q) for the URDF
    velocity = pink.solve_ik(configuration, tasks_to_solve, dt=0.02, solver="quadprog")
    configuration.integrate_inplace(velocity, 0.02)
    
    # E. Save the Result
    # This 'q' is a list of angles that make the robot match the human
    retargeted_data.append(configuration.q.copy())