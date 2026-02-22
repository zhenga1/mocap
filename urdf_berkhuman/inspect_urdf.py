###
## This doesn't work 
###
from urdfpy import URDF
import sys

def print_tree(robot, root, indent=""):
    """Recursively print the kinematic tree."""
    for joint in robot.joints:
        if joint.parent == root:
            print(f"{indent}└──({joint.name})──> {joint.child}")
            print_tree(robot, joint.child, indent + "    ")

def main(urdf_path):
    ### BUGGING
    robot = URDF.load(urdf_path)

    print("\n=== LINKS ===")
    for link in robot.links:
        print(f"- {link.name}")

    print("\n=== JOINTS ===")
    for joint in robot.joints:
        print(f"- {joint.name} ({joint.joint_type})")
        print(f"    Parent: {joint.parent}")
        print(f"    Child : {joint.child}")

    print("\n=== KINEMATIC TREE ===")

    # Find root link (link that is never a child)
    child_links = {joint.child for joint in robot.joints}
    root_links = [link.name for link in robot.links if link.name not in child_links]

    if not root_links:
        print("Could not determine root link.")
        return

    root = root_links[0]
    print(root)
    print_tree(robot, root)

if __name__ == "__main__":
    urdf_path = "C:\\Users\\aaron\\IsaacSim_4.0.0\\Berkeley-Humanoid-Lite\\source\\berkeley_humanoid_lite_assets\\data\\robots\\berkeley_humanoid\\berkeley_humanoid_lite\\urdf\\berkeley_humanoid_lite.urdf"

    main(urdf_path)
