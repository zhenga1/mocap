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

# Reuse the world-coordinate mapping that we already validated in
# left_right_foot_pelvis so both retargeters share exactly the same
# CMU->robot axis conventions and scale.
try:
    from scripts.retargeting.left_right_foot_pelvis import get_scaled_target as _lr_get_scaled_target
except ImportError:
    from left_right_foot_pelvis import get_scaled_target as _lr_get_scaled_target

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
DT = 0.02
POSITION_COST = 1.0
ORIENTATION_COST = 0.1
# Smoothing for noisy CMU targets before IK (offline, so bidirectional is fine).
TARGET_SMOOTH_ALPHA = 0.2
TARGET_SMOOTH_PASSES = 2
# Root trajectory smoothing (saved and replayed as world offset).
ROOT_SMOOTH_ALPHA = 0.25
# Safety clamp for per-frame configuration jumps (radians for joints).
MAX_Q_STEP = 0.08
# Optional post-IK q smoothing. Keep off by default to avoid over-damping.
POST_Q_SMOOTH_ALPHA = 0.0
YAW_SMOOTH_ALPHA = 0.2
SHOULDER_POSITION_COST = 0.3
SHOULDER_TARGET_MAX_ERROR = 0.15
SHOULDER_FRAMES = {"arm_left_shoulder_pitch", "arm_right_shoulder_pitch"}

HEADING_LEFT_RIGHT_CANDIDATES = [
    ("lclavicle", "rclavicle"),
    ("lshoulder", "rshoulder"),
    ("lhipjoint", "rhipjoint"),
    ("lhip", "rhip"),
]


def _to_robot_world(human_coordinate, init_root):
    """
    Human [3,1] -> robot *world* coordinates using the exact same mapping
    as left_right_foot_pelvis.get_scaled_target (CMU Y-up -> robot Z-up,
    global scale, vertical offset).
    """
    return _lr_get_scaled_target(human_coordinate, init_root)


def _to_robot_pos_relative(joint_coordinate, root_coordinate, init_root):
    """
    Human [3,1] -> robot frame with:
      - coordinates made relative to the initial pelvis/root position
      - world mapping shared with left_right_foot_pelvis

    We first map both the joint and the root to the robot *world* frame using
    _to_robot_world, then express the joint position in the root's local frame.
    """
    joint_world = _to_robot_world(joint_coordinate, init_root)
    root_world = _to_robot_world(root_coordinate, init_root)
    return joint_world - root_world


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
        position_cost = SHOULDER_POSITION_COST if robot_frame in SHOULDER_FRAMES else POSITION_COST
        tasks.append(FrameTask(robot_frame, position_cost=position_cost, orientation_cost=ORIENTATION_COST))
        task_cmu_names.append(cmu_name)
    return tasks, task_cmu_names


def _smooth_exponential(data, alpha):
    """
    Simple per-dimension exponential moving average over time:
        y[t] = alpha * x[t] + (1 - alpha) * y[t-1]
    data: (T, D) array.
    """
    if data.size == 0 or alpha <= 0.0:
        return data
    smoothed = np.empty_like(data)
    smoothed[0] = data[0]
    for t in range(1, data.shape[0]):
        smoothed[t] = alpha * data[t] + (1.0 - alpha) * smoothed[t - 1]
    return smoothed


def _smooth_bidirectional_exponential(data, alpha, passes=1):
    """
    Zero-phase-like smoothing by running EMA forward and backward.
    data: (T, D) array.
    """
    if data.size == 0 or alpha <= 0.0:
        return data
    out = data.copy()
    for _ in range(max(1, int(passes))):
        out = _smooth_exponential(out, alpha)
        out = _smooth_exponential(out[::-1], alpha)[::-1]
    return out

def _wrap_to_pi(angle):
    return (angle + np.pi) % (2.0 * np.pi) - np.pi

def _rotation_z(yaw):
    c = np.cos(yaw)
    s = np.sin(yaw)
    return np.array(
        [[c, -s, 0.0],
         [s, c, 0.0],
         [0.0, 0.0, 1.0]],
        dtype=float,
    )

def _estimate_heading_yaw_from_joints(joints, init_root):
    """Estimate heading yaw from left-right body landmarks in robot world frame."""
    for left_name, right_name in HEADING_LEFT_RIGHT_CANDIDATES:
        if left_name not in joints or right_name not in joints:
            continue
        left_world = _to_robot_world(joints[left_name].coordinate, init_root)
        right_world = _to_robot_world(joints[right_name].coordinate, init_root)
        lateral = left_world - right_world
        lateral_xy = lateral[:2]
        norm = np.linalg.norm(lateral_xy)
        if norm < 1e-6:
            continue
        lateral_xy = lateral_xy / norm
        forward_xy = np.array([-lateral_xy[1], lateral_xy[0]], dtype=float)
        return float(np.arctan2(forward_xy[1], forward_xy[0]))
    return 0.0

def _relax_unreachable_target(current_pos, target_pos, max_error):
    delta = target_pos - current_pos
    err = float(np.linalg.norm(delta))
    if err <= max_error or err <= 1e-9:
        return target_pos
    return current_pos + (max_error / err) * delta


def retarget_motion(asf_path, amc_path, robot, configuration, tasks, task_cmu_names):
    """Retarget one AMC with one ASF; returns (T, nq), (T, 3) root position, (T, 1) root yaw."""
    joints = amc.parse_asf(asf_path)
    motions = amc.parse_amc(amc_path)
    if not motions:
        return np.zeros((0, robot.model.nq)), np.zeros((0, 3)), np.zeros((0, 1))

    # Use the pelvis/root position from the first frame as the reference for all
    # world positions, so both human and robot move in a comparable frame.
    joints["root"].set_motion(motions[0])
    init_root = joints["root"].coordinate.copy()

    n_frames = len(motions)
    n_tasks = len(tasks)
    raw_targets = np.zeros((n_frames, n_tasks, 3), dtype=float)
    root_positions = np.zeros((n_frames, 3), dtype=float)
    root_yaws = np.zeros((n_frames, 1), dtype=float)

    for fi, frame in enumerate(motions):
        joints["root"].set_motion(frame)
        root_pos = joints["root"].coordinate.copy()
        root_positions[fi] = _to_robot_world(root_pos, init_root)

        yaw = _estimate_heading_yaw_from_joints(joints, init_root)
        root_yaws[fi, 0] = yaw

        for ti, cmu_name in enumerate(task_cmu_names):
            pos = joints[cmu_name].coordinate
            # Keep IK targets in root-relative world frame; heading is applied in replay.
            target_world_rel = _to_robot_pos_relative(pos, root_pos, init_root)
            raw_targets[fi, ti] = target_world_rel

    smoothed_targets = raw_targets.copy()
    if TARGET_SMOOTH_ALPHA > 0.0:
        for ti in range(n_tasks):
            smoothed_targets[:, ti, :] = _smooth_bidirectional_exponential(
                smoothed_targets[:, ti, :],
                TARGET_SMOOTH_ALPHA,
                passes=TARGET_SMOOTH_PASSES,
            )
    if ROOT_SMOOTH_ALPHA > 0.0:
        root_positions = _smooth_bidirectional_exponential(
            root_positions,
            ROOT_SMOOTH_ALPHA,
            passes=TARGET_SMOOTH_PASSES,
        )
    if YAW_SMOOTH_ALPHA > 0.0:
        yaw_unwrapped = np.unwrap(root_yaws[:, 0])
        yaw_smoothed = _smooth_bidirectional_exponential(
            yaw_unwrapped.reshape(-1, 1),
            YAW_SMOOTH_ALPHA,
            passes=TARGET_SMOOTH_PASSES,
        )[:, 0]
        root_yaws[:, 0] = _wrap_to_pi(yaw_smoothed)

    configuration.q = robot.q0.copy()
    retargeted = []
    for fi in range(n_frames):
        pin.forwardKinematics(robot.model, robot.data, configuration.q)
        pin.updateFramePlacements(robot.model, robot.data)

        # Effective targets actually sent to the IK solver this frame
        frame_targets = np.zeros((n_tasks, 3), dtype=float)
        for ti, task in enumerate(tasks):
            target = smoothed_targets[fi, ti]
            # if task.frame in SHOULDER_FRAMES:
            frame_id = robot.model.getFrameId(task.frame)
            current = robot.data.oMf[frame_id].translation
            target = _relax_unreachable_target(current, target, SHOULDER_TARGET_MAX_ERROR)
            frame_targets[ti] = target
            task.set_target(pin.SE3(np.eye(3), target))
        q_prev = configuration.q.copy()
        velocity = pink.solve_ik(configuration, tasks, dt=DT, solver="quadprog")
        # Update the configuration of the robot using the velocity with timestep DT.
        # If the solver returns non-finite values, treat it as a hard failure for this frame.
        if not np.all(np.isfinite(velocity)):
            # just use the previous configuration (i.e. no change)
            configuration.q = q_prev
        else:
            configuration.integrate_inplace(velocity, DT)
        if MAX_Q_STEP > 0.0:
            q_next = configuration.q.copy()
            dq = np.clip(q_next - q_prev, -MAX_Q_STEP, MAX_Q_STEP)
            configuration.q = q_prev + dq

        # Check for IK "failure" by evaluating achieved vs requested frame positions.
        # (Pink may not raise on infeasible tasks; this catches unreachable targets.)
        pin.forwardKinematics(robot.model, robot.data, configuration.q)
        pin.updateFramePlacements(robot.model, robot.data)

        worst_err = 0.0
        worst_frame = None
        for ti, task in enumerate(tasks):
            frame_id = robot.model.getFrameId(task.frame)
            current_pos = robot.data.oMf[frame_id].translation
            target_pos = frame_targets[ti]
            err = float(np.linalg.norm(current_pos - target_pos))
            if err > worst_err:
                worst_err = err
                worst_frame = task.frame

        # For shoulders we already "relaxed" unreachable targets to within
        # SHOULDER_TARGET_MAX_ERROR of the current pose, so use that as a looser
        # failure threshold; keep 5cm for the rest of the body.
        failure_threshold = 0.05
        if worst_frame in SHOULDER_FRAMES:
            failure_threshold = SHOULDER_TARGET_MAX_ERROR

        if worst_err > failure_threshold:
            print(f"IK FAILURE @ frame {fi}: worst {worst_frame} error {worst_err:.4f}m (target likely unreachable)")
        retargeted.append(configuration.q.copy())

    q_array = np.array(retargeted)
    if POST_Q_SMOOTH_ALPHA > 0.0:
        q_array = _smooth_bidirectional_exponential(
            q_array,
            POST_Q_SMOOTH_ALPHA,
            passes=TARGET_SMOOTH_PASSES,
        )
    return q_array, root_positions, root_yaws

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
    q_trajectory, root_positions, root_yaws = retarget_motion(asf_path, amc_path, robot, configuration, tasks, task_cmu_names)
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        # Save q + root position + root yaw so replay can move and rotate pelvis in world.
        trajectory_with_root = np.hstack([q_trajectory, root_positions, root_yaws])
        np.save(save_path, trajectory_with_root)
        print(f"Saved {save_path} ({q_trajectory.shape[0]} frames, {len(tasks)} tasks, root trajectory included)")
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
                q_trajectory, root_positions, root_yaws = retarget_motion(asf_path, amc_path, robot, configuration, tasks, task_cmu_names)
                trajectory_with_root = np.hstack([q_trajectory, root_positions, root_yaws])
                np.save(save_path, trajectory_with_root)
                print(f"Saved {save_path} ({q_trajectory.shape[0]} frames, {len(tasks)} tasks, root trajectory included)")
            except Exception as e:
                print(f"Skip {asf_path} + {amc_path}: {e}")


if __name__ == "__main__":
    if len(sys.argv) >= 3:
        asf_path, amc_path = sys.argv[1], sys.argv[2]
        save_path = sys.argv[3] if len(sys.argv) > 3 else None
        run_single(asf_path, amc_path, save_path)
    else:
        run_batch()

