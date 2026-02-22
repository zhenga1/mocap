"""
Motion retargeting: CMU MoCap (ASF/AMC) -> Berkeley Humanoid joint angles.

Pipeline (per frame):
  1. Update human skeleton from AMC frame (forward kinematics in amc_parser).
  2. Read world positions of root (pelvis), lfoot, rfoot from the ASF skeleton.
  3. Map human coords to robot coords: CMU uses Y-up; many robots use Z-up, so we
     send (human_x, human_z, human_y) as (robot_x, robot_y, robot_z) and scale.
  4. Set Pink frame tasks for base and both feet to those positions (identity orientation).
  5. Solve IK with Pink (velocity-based, quadprog); integrate to get new q.
  6. Append q; optionally save as .npy.

Correctness:
  - ASF/AMC: amc_parser computes joint coordinates in world frame via set_motion()
    (root translation + rotation, then child positions from hierarchy and bone lengths).
  - Coordinate mapping: human [x,y,z] -> robot [x,z,y] matches Y-up -> Z-up and preserves
    left/right and forward/back; SCALE (0.45) brings human scale to robot scale.
  - Pink IK: FrameTask targets are in world frame; solve_ik finds a velocity that
    moves task frames toward targets; integrate_inplace updates q so the next frame
    starts from the previous solution (temporal continuity).
  - Batch: same robot and tasks reused; configuration is reset per motion via
    configuration.q = robot.q0 at the start of retarget_motion.
"""
import amc_parser as amc
import pinocchio as pin
import numpy as np
import pink
import os
from get_motion_registry import get_motion_registry
from pink.tasks import FrameTask

# Berkeley Humanoid URDF (adjust path if needed)
URDF_PATH = os.path.join(
    os.environ.get("BERKELEY_URDF", ""),
    "C:\\Users\\aaron\\IsaacSim_4.0.0\\Berkeley-Humanoid-Lite\\source\\berkeley_humanoid_lite_assets\\data\\robots\\berkeley_humanoid\\berkeley_humanoid_lite\\urdf\\berkeley_humanoid_lite.urdf"
)
if not os.path.isfile(URDF_PATH):
    URDF_PATH = "C:\\Users\\aaron\\IsaacSim_4.0.0\\Berkeley-Humanoid-Lite\\source\\berkeley_humanoid_lite_assets\\data\\robots\\berkeley_humanoid\\berkeley_humanoid_lite\\urdf\\berkeley_humanoid_lite.urdf"

SUBJECTS_DIR = "subjects"
OUTPUT_DIR = "retargeted"
SCALE = 0.45
DT = 0.02


def _to_robot_pos(coordinate):
    """Human [3,1] -> robot frame: (x, y, z) -> (x, z, y) and scale."""
    x, y, z = coordinate[0, 0], coordinate[1, 0], coordinate[2, 0]
    return np.array([x, z, y], dtype=float) * SCALE


def retarget_motion(asf_path, amc_path, robot, configuration, pelvis_task, left_foot_task, right_foot_task):
    """Retarget one AMC with one ASF; returns list of q vectors."""
    joints = amc.parse_asf(asf_path)
    motions = amc.parse_amc(amc_path)
    configuration.q = robot.q0.copy()
    retargeted = []
    for frame in motions:
        joints["root"].set_motion(frame)
        human_lfoot = joints["lfoot"].coordinate
        human_rfoot = joints["rfoot"].coordinate
        human_pelvis = joints["root"].coordinate
        left_foot_task.set_target(pin.SE3(np.eye(3), _to_robot_pos(human_lfoot)))
        right_foot_task.set_target(pin.SE3(np.eye(3), _to_robot_pos(human_rfoot)))
        pelvis_task.set_target(pin.SE3(np.eye(3), _to_robot_pos(human_pelvis)))
        tasks = [pelvis_task, left_foot_task, right_foot_task]
        velocity = pink.solve_ik(configuration, tasks, dt=DT, solver="quadprog")
        configuration.integrate_inplace(velocity, DT)
        retargeted.append(configuration.q.copy())
    return np.array(retargeted)


def run_single(asf_path, amc_path, save_path=None):
    """Run retargeting for one (asf, amc) pair. Optionally save to save_path."""
    urdf_dir = os.path.dirname(os.path.abspath(URDF_PATH))
    robot = pin.RobotWrapper.BuildFromURDF(URDF_PATH, package_dirs=[urdf_dir, os.path.dirname(urdf_dir)])
    pelvis_task = FrameTask("base", position_cost=1.0, orientation_cost=0.1)
    left_foot_task = FrameTask("leg_left_ankle_roll", position_cost=1.0, orientation_cost=0.1)
    right_foot_task = FrameTask("leg_right_ankle_roll", position_cost=1.0, orientation_cost=0.1)
    configuration = pink.Configuration(robot.model, robot.data, robot.q0)
    q_trajectory = retarget_motion(asf_path, amc_path, robot, configuration, pelvis_task, left_foot_task, right_foot_task)
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        np.save(save_path, q_trajectory)
        print(f"Saved {save_path} ({q_trajectory.shape[0]} frames)")
    return q_trajectory


def run_batch(subjects_dir=SUBJECTS_DIR, output_dir=OUTPUT_DIR):
    """Retarget all (asf, amc) pairs under subjects_dir; save .npy under output_dir."""
    registry = get_motion_registry(subjects_dir)
    urdf_dir = os.path.dirname(os.path.abspath(URDF_PATH))
    robot = pin.RobotWrapper.BuildFromURDF(URDF_PATH, package_dirs=[urdf_dir, os.path.dirname(urdf_dir)])
    pelvis_task = FrameTask("base", position_cost=1.0, orientation_cost=0.1)
    left_foot_task = FrameTask("leg_left_ankle_roll", position_cost=1.0, orientation_cost=0.1)
    right_foot_task = FrameTask("leg_right_ankle_roll", position_cost=1.0, orientation_cost=0.1)
    configuration = pink.Configuration(robot.model, robot.data, robot.q0)
    os.makedirs(output_dir, exist_ok=True)
    for asf_path, amc_list in registry.items():
        if not amc_list:
            continue
        for amc_path in amc_list:
            subject = os.path.splitext(os.path.basename(asf_path))[0]
            motion_name = os.path.splitext(os.path.basename(amc_path))[0]
            out_name = f"{subject}_{motion_name}.npy"
            save_path = os.path.join(output_dir, out_name)
            try:
                q_trajectory = retarget_motion(asf_path, amc_path, robot, configuration, pelvis_task, left_foot_task, right_foot_task)
                np.save(save_path, q_trajectory)
                print(f"Saved {save_path} ({q_trajectory.shape[0]} frames)")
            except Exception as e:
                print(f"Skip {asf_path} + {amc_path}: {e}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 3:
        # Single: python retargeting.py <asf> <amc> [out.npy]
        asf_path, amc_path = sys.argv[1], sys.argv[2]
        save_path = sys.argv[3] if len(sys.argv) > 3 else None
        run_single(asf_path, amc_path, save_path)
    else:
        # Batch: all subjects
        run_batch()