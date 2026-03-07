import os
import sys

_r = os.path.dirname(os.path.abspath(__file__))
while _r and not os.path.isfile(os.path.join(_r, "amc_parser.py")):
    _r = os.path.dirname(_r)
if _r and _r not in sys.path:
    sys.path.insert(0, _r)

import amc_parser as amc
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import numpy as np

def animate_mocap(asf_path, amc_path):
    joints = amc.parse_asf(asf_path)
    motions = amc.parse_amc(amc_path)
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Lines (bones) will be recreated each frame without clearing the axes.
    lines = []

    # Scatter for joints; we update its data each frame and attach CMU joint names.
    scatter = ax.scatter([], [], [], c='blue', s=50, alpha=0.9, picker=True)
    scatter._joint_names = []  # type: ignore[attr-defined]

    # Root trajectory
    traj_line, = ax.plot([], [], [], color='red', alpha=0.5, linewidth=1)
    root_x, root_y, root_z = [], [], []

    # Annotation text for clicked joint name
    annotation = ax.text2D(
        0.02, 0.98, "", transform=ax.transAxes,
        ha="left", va="top", fontsize=10, color="black"
    )

    def update(frame_idx):
        frame = motions[frame_idx]
        joints['root'].set_motion(frame)

        all_x, all_y, all_z = [], [], []

        # Remove old bone lines
        for line in lines:
            line.remove()
        lines.clear()

        # Draw bones (parent-child segments)
        for name, joint in joints.items():
            if joint.parent:
                p1 = joint.parent.coordinate
                p2 = joint.coordinate
                x_data = [p1[0, 0], p2[0, 0]]
                # Swap y and z to use Z-up
                z_data = [p1[1, 0], p2[1, 0]]
                y_data = [p1[2, 0], p2[2, 0]]

                all_x.extend(x_data)
                all_y.extend(y_data)
                all_z.extend(z_data)

                line, = ax.plot(x_data, y_data, z_data, 'b-', linewidth=2)
                lines.append(line)

        # Update joint scatter (one point per CMU joint)
        joint_names = []
        xs, ys, zs = [], [], []
        # Sort by name to keep a deterministic ordering
        for name in sorted(joints.keys()):
            joint = joints[name]
            pos = joint.coordinate
            x, y, z = pos[0, 0], pos[2, 0], pos[1, 0]  # swap y/z
            xs.append(x)
            ys.append(y)
            zs.append(z)
            joint_names.append(name)
            all_x.append(x)
            all_y.append(y)
            all_z.append(z)

        scatter._offsets3d = (xs, ys, zs)
        scatter._joint_names = joint_names  # type: ignore[attr-defined]

        # Update root trajectory
        pos_root = joints['root'].coordinate
        rx, ry, rz = pos_root[0, 0], pos_root[2, 0], pos_root[1, 0]
        root_x.append(rx)
        root_y.append(ry)
        root_z.append(rz)
        traj_line.set_data(root_x, root_y)
        traj_line.set_3d_properties(root_z)

        # Axis limits
        if all_x:
            margin = 20
            ax.set_xlim3d([min(all_x) - margin, max(all_x) + margin])
            ax.set_ylim3d([min(all_y) - margin, max(all_y) + margin])
            ax.set_zlim3d([min(all_z) - margin, max(all_z) + margin])

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'Frame {frame_idx + 1}/{len(motions)}')

        return lines + [scatter, traj_line]

    def on_pick(event):
        """Display CMU joint name when a joint point is clicked."""
        artist = event.artist
        if not hasattr(artist, "_joint_names"):
            return
        ind = getattr(event, "ind", None)
        if ind is None or len(ind) == 0:
            return
        idx = ind[0]
        joint_names = artist._joint_names  # type: ignore[attr-defined]
        if idx < 0 or idx >= len(joint_names):
            return
        name = joint_names[idx]
        annotation.set_text(f"Joint: {name}")
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("pick_event", on_pick)

    # Start animation
    ani = FuncAnimation(fig, update, frames=len(motions), interval=5, repeat=True)
    
    plt.show()

# Run it
animate_mocap('subjects/01/01.asf', 'subjects/01/01_05.amc')
#animate_mocap('subjects/103/103.asf', 'subjects/103/103_01.amc')