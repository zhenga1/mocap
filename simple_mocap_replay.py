import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import amc_parser as amc # Assuming you have the parser file in your folder

def visualize_flip(asf_path, amc_path):
    joints = amc.parse_asf(asf_path)
    motions = amc.parse_amc(amc_path)
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Trajectory storage
    root_x, root_y, root_z = [], [], []

    for frame in motions:
        ax.cla() # Clear for animation
        
        # Apply the motion to the skeleton
        joints['root'].set_motion(frame)
        
        # Track the 'root' (pelvis) trajectory
        # This is how you identify a real jump vs a floor flip
        root_pos = joints['root'].coordinate
        root_x.append(root_pos[0])
        root_y.append(root_pos[1])
        root_z.append(root_pos[2])

        # Draw the skeleton
        joints['root'].draw()
        
        # Draw the trajectory so far
        ax.plot(root_x, root_y, root_z, color='red', alpha=0.5)
        
        # Fix the view so it doesn't jump around
        ax.set_xlim3d([-50, 50])
        ax.set_ylim3d([-50, 50])
        ax.set_zlim3d([0, 100])
        
        plt.pause(0.001) # High-speed replay

# Example: Run it on a flip folder
visualize_flip('subjects/103/103.asf', 'subjects/103/103_01.amc')