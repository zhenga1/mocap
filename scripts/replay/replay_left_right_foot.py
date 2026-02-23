"""
Replay retargeted .npy motion on the Berkeley Humanoid (full skeleton).

This is the retargeted motion with only the left and right foot tasks and the pelvis task.
Uses Pinocchio's frame tree: every frame position is computed from FK, then
each frame is connected to its parent, giving a full stick-figure skeleton.

Usage:
  python replay_left_right_foot.py ../retargeted_only_pelvis_left_right_foot/01_01_05.npy
  python replay_left_right_foot.py ../retargeted_only_pelvis_left_right_foot/01_01_05.npy ../retargeted_only_pelvis_left_right_foot/01_01_01.npy
  python replay_left_right_foot.py   # default: ../retargeted_only_pelvis_left_right_foot/01_01_01.npy
"""
import os
import sys

_r = os.path.dirname(os.path.abspath(__file__))
while _r and not os.path.isfile(os.path.join(_r, "amc_parser.py")):
    _r = os.path.dirname(_r)
if _r and _r not in sys.path:
    sys.path.insert(0, _r)

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import pinocchio as pin

try:
    from retargeting import URDF_PATH, OUTPUT_DIR
except ImportError:
    URDF_PATH = os.path.join(
        os.environ.get("BERKELEY_URDF", ""),
        "C:\\Users\\aaron\\IsaacSim_4.0.0\\Berkeley-Humanoid-Lite\\source\\berkeley_humanoid_lite_assets\\data\\robots\\berkeley_humanoid\\berkeley_humanoid_lite\\urdf\\berkeley_humanoid_lite.urdf"
    )
    if not os.path.isfile(URDF_PATH):
        URDF_PATH = "C:\\Users\\aaron\\IsaacSim_4.0.0\\Berkeley-Humanoid-Lite\\source\\berkeley_humanoid_lite_assets\\data\\robots\\berkeley_humanoid\\berkeley_humanoid_lite\\urdf\\berkeley_humanoid_lite.urdf"
    OUTPUT_DIR = "../retargeted_only_pelvis_left_right_foot"


def load_robot():
    urdf_dir = os.path.dirname(os.path.abspath(URDF_PATH))
    robot = pin.RobotWrapper.BuildFromURDF(
        URDF_PATH,
        package_dirs=[urdf_dir, os.path.dirname(urdf_dir)]
    )
    return robot


def build_skeleton_edges(model):
    """Return list of (child_frame_id, parent_frame_id) for all frames with a non-world parent."""
    edges = []
    for i in range(1, len(model.frames)):
        frame = model.frames[i]
        parent = getattr(frame, "parentFrame", getattr(frame, "previousFrame", -1))
        if parent < 0 and hasattr(frame, "parent"):
            parent = frame.parent  # some bindings use .parent for parent frame
        if parent >= 0:
            edges.append((i, parent))
    return edges


def get_all_frame_positions(robot, q):
    """Return (positions dict by frame index, skeleton edges)."""
    pin.forwardKinematics(robot.model, robot.data, q)
    pin.updateFramePlacements(robot.model, robot.data)
    positions = {}
    for i in range(len(robot.model.frames)):
        positions[i] = robot.data.oMf[i].translation.copy()
    return positions


def animate_retargeted(npy_paths, interval=50):
    """Animate one or more retargeted .npy files; draw full skeleton + pelvis trajectory."""
    if not npy_paths:
        npy_paths = [os.path.join(OUTPUT_DIR, "01_01_01.npy")]
    robot = load_robot()
    model = robot.model
    skeleton_edges = build_skeleton_edges(model)
    base_fid = model.getFrameId("base") if model.existFrame("base") else 0

    all_q = []
    for path in npy_paths:
        if not os.path.isfile(path):
            print(f"Skip (not found): {path}")
            continue
        q = np.load(path)
        if q.ndim == 1:
            q = q.reshape(1, -1)
        all_q.append((path, q))
    if not all_q:
        raise FileNotFoundError("No .npy files found.")

    q_trajectories = [q for _, q in all_q]
    total_frames = sum(q.shape[0] for q in q_trajectories)
    frame_to_file_and_row = []
    for fi, q in enumerate(q_trajectories):
        for row in range(q.shape[0]):
            frame_to_file_and_row.append((fi, row))

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    pelvis_traj_x, pelvis_traj_y, pelvis_traj_z = [], [], []

    def update(frame_idx):
        ax.clear()
        if frame_idx == 0:
            pelvis_traj_x.clear()
            pelvis_traj_y.clear()
            pelvis_traj_z.clear()
        fi, row = frame_to_file_and_row[frame_idx]
        q = q_trajectories[fi][row]
        positions = get_all_frame_positions(robot, q)

        all_x, all_y, all_z = [], [], []
        for (child, parent) in skeleton_edges:
            if child not in positions or parent not in positions:
                continue
            pc, pp = positions[child], positions[parent]
            ax.plot([pc[0], pp[0]], [pc[1], pp[1]], [pc[2], pp[2]], "b-", linewidth=1.5)
            all_x.extend([pc[0], pp[0]])
            all_y.extend([pc[1], pp[1]])
            all_z.extend([pc[2], pp[2]])
        xs = [positions[i][0] for i in positions]
        ys = [positions[i][1] for i in positions]
        zs = [positions[i][2] for i in positions]
        ax.scatter(xs, ys, zs, c="blue", s=25, alpha=0.9)

        if base_fid >= 0 and base_fid in positions:
            p = positions[base_fid]
            pelvis_traj_x.append(p[0])
            pelvis_traj_y.append(p[1])
            pelvis_traj_z.append(p[2])
            if len(pelvis_traj_x) > 1:
                ax.plot(pelvis_traj_x, pelvis_traj_y, pelvis_traj_z, color="red", alpha=0.6, linewidth=1)

        if all_x:
            margin = 0.1
            ax.set_xlim3d(min(all_x) - margin, max(all_x) + margin)
            ax.set_ylim3d(min(all_y) - margin, max(all_y) + margin)
            ax.set_zlim3d(max(0, min(all_z) - margin), max(all_z) + margin)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        file_name = os.path.basename(npy_paths[fi])
        ax.set_title(f"Retargeted replay: {file_name}  Frame {frame_idx + 1}/{total_frames}")

    ani = FuncAnimation(fig, update, frames=total_frames, interval=interval, repeat=True)
    plt.show()


if __name__ == "__main__":
    npy_paths = sys.argv[1:] if len(sys.argv) >= 2 else [os.path.join(OUTPUT_DIR, "01_01_01.npy")]
    animate_retargeted(npy_paths)
