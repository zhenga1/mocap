"""
Full-body motion retargeting: CMU MoCap (ASF/AMC) -> Berkeley Humanoid.

Maps as many Berkeley Humanoid frames as possible to CMU skeleton joints
(see cmu_to_berkeley_mapping.py). For each frame we set a position target from
the corresponding human joint, then solve IK with Pink. Tasks are built
dynamically per ASF so subjects with different joint names still work.
"""
import sys
import os

# Ensure project root (directory containing amc_parser.py) is on path
_r = os.path.dirname(os.path.abspath(__file__))
while _r and not os.path.isfile(os.path.join(_r, "amc_parser.py")):
     # get the parent directory, return None if no parent directory
    _r = os.path.dirname(_r)
if _r and _r not in sys.path:
    sys.path.insert(0, _r)
PROJECT_ROOT = _r or os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import amc_parser as amc
import pinocchio as pin
import numpy as np
import pink
from pink.tasks import FrameTask

# Load mapping from same directory (works from project root or scripts/retargeting)
try:
    from scripts.retargeting.cmu_to_berkeley_mapping import BERKELEY_TO_CMU, BERKELEY_TO_CMU_ALT
except ImportError:
    from cmu_to_berkeley_mapping import BERKELEY_TO_CMU, BERKELEY_TO_CMU_ALT

# URDF path
URDF_PATH = os.path.join(
    os.environ.get("BERKELEY_URDF", ""),
    "C:\\Users\\aaron\\IsaacSim_4.0.0\\Berkeley-Humanoid-Lite\\source\\berkeley_humanoid_lite_assets\\data\\robots\\berkeley_humanoid\\berkeley_humanoid_lite\\urdf\\berkeley_humanoid_lite.urdf"
)
if not os.path.isfile(URDF_PATH):
    URDF_PATH = "C:\\Users\\aaron\\IsaacSim_4.0.0\\Berkeley-Humanoid-Lite\\source\\berkeley_humanoid_lite_assets\\data\\robots\\berkeley_humanoid\\berkeley_humanoid_lite\\urdf\\berkeley_humanoid_lite.urdf"

SUBJECTS_DIR = os.path.join(PROJECT_ROOT, "subjects")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "retargeted_full_body")
SCALE = 0.5
DT = 0.02
POSITION_COST = 1.0
ORIENTATION_COST = 0.1


def _to_robot_pos_relative(human_coordinate, init_root):
    """
    Human [3,1] -> robot frame with:
      - coordinates made relative to the initial pelvis/root position
      - CMU Y-up -> robot Z-up axis swap
      - global scale + small vertical offset for better stance.
    This mirrors the logic in left_right_foot_pelvis.get_scaled_target so
    both retargeters use a consistent world mapping.
    """
    # subtracts the initial root position from the human coordinate
    rel = human_coordinate - init_root  # 3x1
    dx, dy, dz = rel[0, 0], rel[1, 0], rel[2, 0]

    # Map CMU (x, y, z) (Y-up) -> robot (x, y, z) (Z-up)
    x_robot = dx * SCALE
    y_robot = -dz * SCALE
    z_robot = dy * SCALE

    return np.array([x_robot, y_robot, z_robot], dtype=float)


def build_tasks_for_skeleton(robot, joints_dict, mapping_list):
    """
    Build (task_list, task_cmu_names) for frames that exist in both robot and ASF.
    joints_dict: from amc.parse_asf(asf_path) (joint name -> Joint).
    mapping_list: list of (robot_frame_name, [cmu_name1, cmu_name2, ...]).
    """
    tasks = []
    task_cmu_names = []  # for each task, the CMU joint name we use
    for robot_frame, cmu_candidates in mapping_list:
        if not robot.model.existFrame(robot_frame):
            continue
        cmu_name = None
        for c in cmu_candidates:
            if c in joints_dict:
                cmu_name = c
                break
        if cmu_name is None:
            continue
        tasks.append(FrameTask(robot_frame, position_cost=POSITION_COST, orientation_cost=ORIENTATION_COST))
        task_cmu_names.append(cmu_name)
    return tasks, task_cmu_names


def retarget_motion(asf_path, amc_path, robot, configuration, tasks, task_cmu_names):
    """Retarget one AMC with one ASF using full-body tasks; returns (T, nq)."""
    joints = amc.parse_asf(asf_path)
    motions = amc.parse_amc(amc_path)
    if not motions:
        return np.zeros((0, robot.model.nq))

    configuration.q = robot.q0.copy()
    retargeted = []

    # Use the pelvis/root position from the first frame as the reference for all
    # world positions, so both human and robot move in a comparable frame.
    joints["root"].set_motion(motions[0])
    # pelvis/root position in everry frame
    init_root = joints["root"].coordinate.copy()

    for frame in motions:
        joints["root"].set_motion(frame)
        # Get current root position in robot frame
        root_pos = joints["root"].coordinate.copy()
        root_robot_pos = _to_robot_pos_relative(root_pos, init_root)
        for task, cmu_name in zip(tasks, task_cmu_names):
            pos = joints[cmu_name].coordinate
            target = _to_robot_pos_relative(pos, init_root) - root_robot_pos
            task.set_target(pin.SE3(np.eye(3), target))
        velocity = pink.solve_ik(configuration, tasks, dt=DT, solver="quadprog")
        configuration.integrate_inplace(velocity, DT)
        retargeted.append(configuration.q.copy())
    return np.array(retargeted)


def run_single(asf_path, amc_path, save_path=None):
    """Run full-body retargeting for one (asf, amc) pair."""
    from get_motion_registry import get_motion_registry
    urdf_dir = os.path.dirname(os.path.abspath(URDF_PATH))
    robot = pin.RobotWrapper.BuildFromURDF(URDF_PATH, package_dirs=[urdf_dir, os.path.dirname(urdf_dir)])
    joints = amc.parse_asf(asf_path)
    mapping = BERKELEY_TO_CMU + BERKELEY_TO_CMU_ALT
    tasks, task_cmu_names = build_tasks_for_skeleton(robot, joints, mapping)
    if not tasks:
        raise RuntimeError("No (robot frame, CMU joint) pairs found for this ASF.")
    configuration = pink.Configuration(robot.model, robot.data, robot.q0)
    q_trajectory = retarget_motion(asf_path, amc_path, robot, configuration, tasks, task_cmu_names)
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        np.save(save_path, q_trajectory)
        print(f"Saved {save_path} ({q_trajectory.shape[0]} frames, {len(tasks)} tasks)")
    return q_trajectory


def run_batch(subjects_dir=SUBJECTS_DIR, output_dir=OUTPUT_DIR):
    """Retarget all (asf, amc) pairs; save .npy under output_dir."""
    from get_motion_registry import get_motion_registry
    registry = get_motion_registry(subjects_dir)
    urdf_dir = os.path.dirname(os.path.abspath(URDF_PATH))
    robot = pin.RobotWrapper.BuildFromURDF(URDF_PATH, package_dirs=[urdf_dir, os.path.dirname(urdf_dir)])
    mapping = BERKELEY_TO_CMU + BERKELEY_TO_CMU_ALT
    os.makedirs(output_dir, exist_ok=True)
    for asf_path, amc_list in registry.items():
        if not amc_list:
            continue
        # below, this give us the available CMU joints for the given ASF file
        joints = amc.parse_asf(asf_path)
        tasks, task_cmu_names = build_tasks_for_skeleton(robot, joints, mapping)
        if not tasks:
            print(f"No tasks for {asf_path}, skip")
            continue
        configuration = pink.Configuration(robot.model, robot.data, robot.q0)
        for amc_path in amc_list:
            subject = os.path.splitext(os.path.basename(asf_path))[0]
            motion_name = os.path.splitext(os.path.basename(amc_path))[0]
            out_name = f"{subject}_{motion_name}.npy"
            save_path = os.path.join(output_dir, out_name)
            try:
                q_trajectory = retarget_motion(asf_path, amc_path, robot, configuration, tasks, task_cmu_names)
                np.save(save_path, q_trajectory)
                print(f"Saved {save_path} ({q_trajectory.shape[0]} frames, {len(tasks)} tasks)")
            except Exception as e:
                print(f"Skip {asf_path} + {amc_path}: {e}")


if __name__ == "__main__":
    if len(sys.argv) >= 3:
        asf_path, amc_path = sys.argv[1], sys.argv[2]
        save_path = sys.argv[3] if len(sys.argv) > 3 else None
        run_single(asf_path, amc_path, save_path)
    else:
        run_batch()
