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
    
    #import pdb; pdb.set_trace()
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Initialize line objects - create empty lines for each bone
    lines = {}
    for name, joint in joints.items():
        if joint.parent:
            lines[name], = ax.plot([], [], [], 'b-', linewidth=2)
    
    traj_line, = ax.plot([], [], [], color='red', alpha=0.5, linewidth=1)
    root_x, root_y, root_z = [], [], []

    def update(frame_idx):
        # Clear previous frame
        ax.clear()
        
        frame = motions[frame_idx]
        joints['root'].set_motion(frame)
        
        # Collect all coordinates for auto-scaling
        all_x, all_y, all_z = [], [], []
        
        # Draw bones
        for name, joint in joints.items():
            if joint.parent:
                p1 = joint.parent.coordinate
                p2 = joint.coordinate
                x_data = [p1[0, 0], p2[0, 0]]
                ## Swap y and z here to match typical 3D coordinate in IsaacLab (Z-up)
                z_data = [p1[1, 0], p2[1, 0]]
                y_data = [p1[2, 0], p2[2, 0]]
                
                all_x.extend(x_data)
                all_y.extend(y_data)
                all_z.extend(z_data)
                
                ax.plot(x_data, y_data, z_data, 'b-', linewidth=2)
        
        # Draw joints as points
        for joint in joints.values():
            pos = joint.coordinate
            ## Swap y and z here to match typical 3D coordinate in IsaacLab (Z-up)
            x_points, y_points, z_points = pos[0, 0], pos[2, 0], pos[1, 0]  # Swap y and z for plotting
            ax.scatter([x_points], [y_points], [z_points], c='blue', s=50)
            all_x.append(x_points)
            all_y.append(y_points)
            all_z.append(z_points)
        
        # Update trajectory
        pos = joints['root'].coordinate
        root_x.append(pos[0, 0])
        ## Swap y and z here to match typical 3D coordinate in IsaacLab (Z-up)
        root_y.append(pos[2, 0])
        root_z.append(pos[1, 0])
        
        if len(root_x) > 1:
            ax.plot(root_x, root_y, root_z, color='red', alpha=0.5, linewidth=1)
        
        # Auto-scale axes with some padding
        if all_x:
            margin = 20
            ax.set_xlim3d([min(all_x) - margin, max(all_x) + margin])
            ax.set_ylim3d([min(all_y) - margin, max(all_y) + margin])
            ax.set_zlim3d([min(all_z) - margin, max(all_z) + margin])
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_zlim3d([0, 50]) # Set a reasonable Z limit for better visualization
        ax.set_title(f'Frame {frame_idx + 1}/{len(motions)}')

    # Start animation
    ani = FuncAnimation(fig, update, frames=len(motions), interval=5, repeat=True)
    
    plt.show()

# Run it
animate_mocap('subjects/01/01.asf', 'subjects/01/01_05.amc')
#animate_mocap('subjects/103/103.asf', 'subjects/103/103_01.amc')