### 
#
# Get the relevant link names within the URDF file
###
import pinocchio as pin

# Load your robot
urdf_path = "C:\\Users\\aaron\\IsaacSim_4.0.0\\Berkeley-Humanoid-Lite\\source\\berkeley_humanoid_lite_assets\\data\\robots\\berkeley_humanoid\\berkeley_humanoid_lite\\urdf\\berkeley_humanoid_lite.urdf"
model = pin.buildModelFromUrdf(urdf_path)

print("--- Robot Links (Frames) ---")
for frame in model.frames:
    print(frame.name)