"""
Replay retargeted .npy motion on the Berkeley Humanoid (full skeleton).

This is the retargeted motion with all the tasks.
Uses Pinocchio's frame tree: every frame position is computed from FK, then
each frame is connected to its parent, giving a full stick-figure skeleton.

Usage:
  python replay_full_body.py ../retargeted_full_body/01_01_05.npy
  python replay_full_body.py ../retargeted_full_body/01_01_05.npy ../retargeted_full_body/01_01_01.npy
  python replay_full_body.py   # default: ../retargeted_full_body/01_01_01.npy
"""
import argparse
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
    # Prefer the full-body retargeting config (URDF path + full-body output dir)
    from scripts.retargeting.full_body import URDF_PATH, OUTPUT_DIR
except ImportError:
    try:
        from retargeting import URDF_PATH, OUTPUT_DIR
        # When importing from the simpler retargeting script, override the output
        # directory so this replay looks at the full-body results by default.
        OUTPUT_DIR = os.path.join(_r, "retargeted_full_body")
    except ImportError:
        # Fallback: explicit URDF path and default full-body output directory.
        URDF_PATH = os.path.join(
            os.environ.get("BERKELEY_URDF", ""),
            "C:\\Users\\aaron\\IsaacSim_4.0.0\\Berkeley-Humanoid-Lite\\source\\berkeley_humanoid_lite_assets\\data\\robots\\berkeley_humanoid\\berkeley_humanoid_lite\\urdf\\berkeley_humanoid_lite.urdf"
        )
        if not os.path.isfile(URDF_PATH):
            URDF_PATH = "C:\\Users\\aaron\\IsaacSim_4.0.0\\Berkeley-Humanoid-Lite\\source\\berkeley_humanoid_lite_assets\\data\\robots\\berkeley_humanoid\\berkeley_humanoid_lite\\urdf\\berkeley_humanoid_lite.urdf"
        OUTPUT_DIR = os.path.join(_r, "retargeted_full_body")


# High-level skeleton frames and edges to visualize a clean humanoid stick figure.
SKELETON_EDGES_BY_NAME = [
    # Left leg
    ("base", "leg_left_hip_roll"),
    ("leg_left_hip_roll", "leg_left_hip_yaw"),
    ("leg_left_hip_yaw", "leg_left_hip_pitch"),
    ("leg_left_hip_pitch", "leg_left_knee_pitch"),
    ("leg_left_knee_pitch", "leg_left_ankle_pitch"),
    ("leg_left_ankle_pitch", "leg_left_ankle_roll"),
    # Right leg
    ("base", "leg_right_hip_roll"),
    ("leg_right_hip_roll", "leg_right_hip_yaw"),
    ("leg_right_hip_yaw", "leg_right_hip_pitch"),
    ("leg_right_hip_pitch", "leg_right_knee_pitch"),
    ("leg_right_knee_pitch", "leg_right_ankle_pitch"),
    ("leg_right_ankle_pitch", "leg_right_ankle_roll"),
    # Left arm
    ("base", "arm_left_shoulder_pitch"),
    ("arm_left_shoulder_pitch", "arm_left_shoulder_roll"),
    ("arm_left_shoulder_roll", "arm_left_shoulder_yaw"),
    ("arm_left_shoulder_yaw", "arm_left_elbow_pitch"),
    ("arm_left_elbow_pitch", "arm_left_elbow_roll"),
    ("arm_left_elbow_roll", "arm_left_hand_link"),
    # Right arm
    ("base", "arm_right_shoulder_pitch"),
    ("arm_right_shoulder_pitch", "arm_right_shoulder_roll"),
    ("arm_right_shoulder_roll", "arm_right_shoulder_yaw"),
    ("arm_right_shoulder_yaw", "arm_right_elbow_pitch"),
    ("arm_right_elbow_pitch", "arm_right_elbow_roll"),
    ("arm_right_elbow_roll", "arm_right_hand_link"),
]


# Replay-time smoothing for already-retargeted trajectories.
REPLAY_SMOOTH_ALPHA = 0.2
REPLAY_SMOOTH_PASSES = 2

def load_robot():
    urdf_dir = os.path.dirname(os.path.abspath(URDF_PATH))
    robot = pin.RobotWrapper.BuildFromURDF(
        URDF_PATH,
        package_dirs=[urdf_dir, os.path.dirname(urdf_dir)]
    )
    return robot


def build_skeleton_edges(model):
    """
    Build a clean humanoid stick-figure by connecting a curated set of
    high-level frames (hips, knees, ankles, shoulders, elbows, hands).

    Returns: list of (child_frame_id, parent_frame_id).
    """
    name_to_id = {}
    for name in {n for edge in SKELETON_EDGES_BY_NAME for n in edge}:
        if model.existFrame(name):
            name_to_id[name] = model.getFrameId(name)

    edges = []
    for child_name, parent_name in SKELETON_EDGES_BY_NAME:
        if child_name in name_to_id and parent_name in name_to_id:
            edges.append((name_to_id[child_name], name_to_id[parent_name]))
    return edges


def get_all_frame_positions(robot, q):
    """Return (positions dict by frame index, skeleton edges)."""
    pin.forwardKinematics(robot.model, robot.data, q)
    pin.updateFramePlacements(robot.model, robot.data)
    positions = {}
    for i in range(len(robot.model.frames)):
        positions[i] = robot.data.oMf[i].translation.copy()
    return positions

def _smooth_exponential(data, alpha):
    if data.size == 0 or alpha <= 0.0:
        return data
    out = np.empty_like(data)
    out[0] = data[0]
    for t in range(1, data.shape[0]):
        out[t] = alpha * data[t] + (1.0 - alpha) * out[t - 1]
    return out

def _smooth_bidirectional_exponential(data, alpha, passes=1):
    if data.size == 0 or alpha <= 0.0:
        return data
    out = data.copy()
    for _ in range(max(1, int(passes))):
        out = _smooth_exponential(out, alpha)
        out = _smooth_exponential(out[::-1], alpha)[::-1]
    return out

def _smooth_replay_data(q, root_pos, root_yaw, alpha, passes):
    if alpha <= 0.0:
        return q, root_pos, root_yaw
    yaw_unwrapped = np.unwrap(root_yaw[:, 0])
    yaw_smoothed = _smooth_bidirectional_exponential(
        yaw_unwrapped.reshape(-1, 1),
        alpha,
        passes=passes,
    )
    return (
        _smooth_bidirectional_exponential(q, alpha, passes=passes),
        _smooth_bidirectional_exponential(root_pos, alpha, passes=passes),
        yaw_smoothed,
    )


def _rotation_z(yaw):
    c = np.cos(yaw)
    s = np.sin(yaw)
    return np.array(
        [[c, -s, 0.0],
         [s, c, 0.0],
         [0.0, 0.0, 1.0]],
        dtype=float,
    )


def animate_retargeted(npy_paths, interval=50, smooth_alpha=REPLAY_SMOOTH_ALPHA, smooth_passes=REPLAY_SMOOTH_PASSES):
    """Animate retargeted .npy files with optional root translation + yaw heading."""
    if not npy_paths:
        npy_paths = [os.path.join(OUTPUT_DIR, "01_01_01.npy")]
    robot = load_robot()
    model = robot.model
    nq = model.nq
    print(model.nq)
    print(model.joints)
    skeleton_edges = build_skeleton_edges(model)
    base_fid = model.getFrameId("base") if model.existFrame("base") else 0

    all_q = []
    for path in npy_paths:
        if not os.path.isfile(path):
            print(f"Skip (not found): {path}")
            continue
        data = np.load(path)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        # New format: (T, nq+4) with root position + yaw; older: (T, nq+3); legacy: (T, nq)
        if data.shape[1] >= nq + 4:
            q = data[:, :nq].copy()
            root_pos = data[:, nq:nq + 3].copy()
            root_yaw = data[:, nq + 3:nq + 4].copy()
        elif data.shape[1] >= nq + 3:
            q = data[:, :nq].copy()
            root_pos = data[:, nq:nq + 3].copy()
            root_yaw = np.zeros((q.shape[0], 1), dtype=float)
        else:
            q = data[:, :nq].copy()
            root_pos = np.zeros((q.shape[0], 3), dtype=float)
            root_yaw = np.zeros((q.shape[0], 1), dtype=float)
        q, root_pos, root_yaw = _smooth_replay_data(q, root_pos, root_yaw, smooth_alpha, smooth_passes)
        all_q.append((path, q, root_pos, root_yaw))
    if not all_q:
        raise FileNotFoundError("No .npy files found.")

    q_trajectories = [q for _, q, _, _ in all_q]
    root_trajectories = [rp for _, _, rp, _ in all_q]
    yaw_trajectories = [ry for _, _, _, ry in all_q]
    paths_loaded = [p for p, _, _, _ in all_q]
    total_frames = sum(q.shape[0] for q in q_trajectories)
    frame_to_file_and_row = []
    for fi, q in enumerate(q_trajectories):
        for row in range(q.shape[0]):
            frame_to_file_and_row.append((fi, row))

    lines = []
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    scatter = ax.scatter([], [], [], c="blue", s=25, alpha=0.9, picker=True)
    scatter._frame_ids = []
    pelvis_traj_x, pelvis_traj_y, pelvis_traj_z = [], [], []
    # Annotation shown when clicking on a vertex
    annotation = ax.text2D(
        0.02,
        0.98,
        "",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        color="black",
    )

    def update(frame_idx):
        if frame_idx == 0:
            pelvis_traj_x.clear()
            pelvis_traj_y.clear()
            pelvis_traj_z.clear()

        fi, row = frame_to_file_and_row[frame_idx]
        q = q_trajectories[fi][row]
        root_offset = root_trajectories[fi][row]
        root_yaw = yaw_trajectories[fi][row, 0]
        rot = _rotation_z(root_yaw)
        positions = get_all_frame_positions(robot, q)
        pivot = positions[base_fid].copy() if base_fid in positions else np.zeros(3, dtype=float)
        # Apply root yaw around the base pivot, then world translation.
        for i in positions:
            positions[i] = rot @ (positions[i] - pivot) + pivot + root_offset

        # Update scatter data
        frame_ids = sorted(positions.keys())
        xs = [positions[i][0] for i in frame_ids]
        ys = [positions[i][1] for i in frame_ids]
        zs = [positions[i][2] for i in frame_ids]

        scatter._offsets3d = (xs, ys, zs)
        scatter._frame_ids = frame_ids

        # We'll use these to set axis limits so the figure doesn't look flattened.
        all_x = xs.copy()
        all_y = ys.copy()
        all_z = zs.copy()

        # Remove old lines
        for line in lines:
            line.remove()
        lines.clear()

        # Draw skeleton edges
        for (child, parent) in skeleton_edges:
            if child not in positions or parent not in positions:
                continue
            pc, pp = positions[child], positions[parent]
            line, = ax.plot(
                [pc[0], pp[0]],
                [pc[1], pp[1]],
                [pc[2], pp[2]],
                "b-",
                linewidth=1.5,
            )
            lines.append(line)
            all_x.extend([pc[0], pp[0]])
            all_y.extend([pc[1], pp[1]])
            all_z.extend([pc[2], pp[2]])

        # Pelvis trajectory
        if base_fid >= 0 and base_fid in positions:
            p = positions[base_fid]
            pelvis_traj_x.append(p[0])
            pelvis_traj_y.append(p[1])
            pelvis_traj_z.append(p[2])
            if len(pelvis_traj_x) > 1:
                traj_line, = ax.plot(
                    pelvis_traj_x,
                    pelvis_traj_y,
                    pelvis_traj_z,
                    color="red",
                    alpha=0.6,
                    linewidth=1,
                )
                lines.append(traj_line)

        # Axis labels and dynamic limits
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        if all_x:
            margin = 0.1
            ax.set_xlim3d(min(all_x) - margin, max(all_x) + margin)
            ax.set_ylim3d(min(all_y) - margin, max(all_y) + margin)
            ax.set_zlim3d(max(0, min(all_z) - margin), max(all_z) + margin)

        # Title with current frame index (like a simple frame slider)
        file_name = os.path.basename(paths_loaded[fi])
        ax.set_title(f"Full-body retargeted replay: {file_name}  Frame {frame_idx + 1}/{total_frames}")
    def on_pick(event):
        """Display the Pinocchio frame name when the user clicks a vertex."""
        artist = event.artist
        # Only handle our main scatter plot
        if not hasattr(artist, "_frame_ids"):
            return
        ind = getattr(event, "ind", None)
        if ind is None or len(ind) == 0:
            return
        idx = ind[0]
        frame_ids = artist._frame_ids  # type: ignore[attr-defined]
        if idx < 0 or idx >= len(frame_ids):
            return
        frame_id = frame_ids[idx]
        frame_name = model.frames[frame_id].name
        annotation.set_text(f"Frame {frame_id}: {frame_name}")
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("pick_event", on_pick)

    ani = FuncAnimation(fig, update, frames=total_frames, interval=interval, repeat=True)
    plt.show()


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--npy_paths", nargs="+", default=[os.path.join(OUTPUT_DIR, "01_01_01.npy")])
    argparser.add_argument("--interval", type=int, default=50)
    argparser.add_argument("--smooth_alpha", type=float, default=REPLAY_SMOOTH_ALPHA)
    argparser.add_argument("--smooth_passes", type=int, default=REPLAY_SMOOTH_PASSES)
    args = argparser.parse_args()
    animate_retargeted(args.npy_paths, args.interval, args.smooth_alpha, args.smooth_passes)