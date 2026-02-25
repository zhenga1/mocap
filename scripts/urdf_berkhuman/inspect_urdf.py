"""
Interactive inspection of the Berkeley Humanoid URDF.

What this script does:
  - Loads the URDF with Pinocchio.
  - Prints all Pinocchio frames and their parent joints.
  - Opens a 3D matplotlib window showing the kinematic tree:
      * Blue points at each frame position for q0.
      * Lines between frames and their parent frames.
      * Key frames (base, hips, IMUs, etc.) are highlighted.
      * Clicking on a point displays the corresponding frame name.
"""

import os

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import numpy as np
import pinocchio as pin


URDF_PATH = (
    "C:\\Users\\aaron\\IsaacSim_4.0.0\\Berkeley-Humanoid-Lite\\source\\"
    "berkeley_humanoid_lite_assets\\data\\robots\\berkeley_humanoid\\"
    "berkeley_humanoid_lite\\urdf\\berkeley_humanoid_lite.urdf"
)


KEY_FRAME_NAMES = [
    "base",
    "leg_left_hip_roll",
    "leg_right_hip_roll",
    "leg_left_ankle_roll",
    "leg_right_ankle_roll",
    "imu",
    "imu_2",
    "imu_frame",
]


def load_robot():
    urdf_dir = os.path.dirname(os.path.abspath(URDF_PATH))
    robot = pin.RobotWrapper.BuildFromURDF(
        URDF_PATH,
        package_dirs=[urdf_dir, os.path.dirname(urdf_dir)],
    )
    return robot


def print_frame_overview(model):
    print("\n=== Pinocchio Frames (index : name | parent joint) ===")
    for fid, frame in enumerate(model.frames):
        parent_joint_name = model.names[frame.parentJoint] if frame.parentJoint < len(
            model.names
        ) else "UNKNOWN"
        print(f"{fid:3d}: {frame.name:30s} | parent joint: {parent_joint_name}")


def get_frame_positions(robot):
    """Return dict frame_id -> 3-vector position for q0."""
    pin.forwardKinematics(robot.model, robot.data, robot.q0)
    pin.updateFramePlacements(robot.model, robot.data)
    positions = {}
    for i in range(len(robot.model.frames)):
        positions[i] = robot.data.oMf[i].translation.copy()
    return positions


def build_edges(model):
    """Return list of (child_frame_id, parent_frame_id) for visualization."""
    edges = []
    for fid in range(1, len(model.frames)):
        frame = model.frames[fid]
        parent = getattr(frame, "parentFrame", getattr(frame, "previousFrame", -1))
        if parent < 0 and hasattr(frame, "parent"):
            parent = frame.parent
        if parent >= 0:
            edges.append((fid, parent))
    return edges


def visualize(robot):
    model = robot.model
    positions = get_frame_positions(robot)
    edges = build_edges(model)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Build arrays for scatter
    frame_ids = sorted(positions.keys())
    xs = [positions[i][0] for i in frame_ids]
    ys = [positions[i][1] for i in frame_ids]
    zs = [positions[i][2] for i in frame_ids]

    scatter = ax.scatter(xs, ys, zs, c="blue", s=30, alpha=0.9, picker=True)
    scatter._frame_ids = frame_ids  # type: ignore[attr-defined]

    # Highlight key frames in red and annotate them.
    # Some names (e.g. "imu") appear multiple times with different FrameTypes,
    # so we resolve them manually by scanning model.frames instead of using
    # getFrameId(name), which can raise a ValueError in that case.
    for name in KEY_FRAME_NAMES:
        # Find all frame ids with this name
        candidate_fids = [i for i, fr in enumerate(model.frames) if fr.name == name]
        if not candidate_fids:
            continue
        # Pick the first match for visualization; print if there are multiple.
        if len(candidate_fids) > 1:
            print(f'[inspect_urdf] Multiple frames named "{name}": {candidate_fids}; using {candidate_fids[0]}')
        fid = candidate_fids[0]
        p = positions[fid]
        ax.scatter([p[0]], [p[1]], [p[2]], c="red", s=60)
        ax.text(p[0], p[1], p[2], f" {name}", color="red")

    # Draw edges
    all_x, all_y, all_z = [], [], []
    for child, parent in edges:
        pc, pp = positions[child], positions[parent]
        ax.plot(
            [pc[0], pp[0]],
            [pc[1], pp[1]],
            [pc[2], pp[2]],
            "k-",
            linewidth=0.7,
            alpha=0.6,
        )
        all_x.extend([pc[0], pp[0]])
        all_y.extend([pc[1], pp[1]])
        all_z.extend([pc[2], pp[2]])

    # Axis labels and limits
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    if all_x:
        margin = 0.1
        ax.set_xlim3d(min(all_x) - margin, max(all_x) + margin)
        ax.set_ylim3d(min(all_y) - margin, max(all_y) + margin)
        ax.set_zlim3d(max(0, min(all_z) - margin), max(all_z) + margin)

    # Annotation updated on pick
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

    def on_pick(event):
        artist = event.artist
        if not hasattr(artist, "_frame_ids"):
            return
        ind = getattr(event, "ind", None)
        if ind is None or len(ind) == 0:
            return
        idx = ind[0]
        frame_ids_local = artist._frame_ids  # type: ignore[attr-defined]
        if idx < 0 or idx >= len(frame_ids_local):
            return
        fid = frame_ids_local[idx]
        frame_name = model.frames[fid].name
        annotation.set_text(f"Frame {fid}: {frame_name}")
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("pick_event", on_pick)

    ax.set_title("Berkeley Humanoid URDF: frame positions and connections (q0)")
    plt.show()


def main():
    robot = load_robot()
    print_frame_overview(robot.model)
    visualize(robot)


if __name__ == "__main__":
    main()
