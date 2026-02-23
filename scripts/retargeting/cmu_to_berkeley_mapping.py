"""
Mapping from Berkeley Humanoid URDF frame names to CMU ASF/AMC joint names.

CMU joint names can vary by subject (e.g. lupperleg vs lfemur). Each robot frame
maps to a list of possible CMU names; we use the first that exists in the parsed ASF.
"""

# Robot frame name -> list of possible CMU joint names (first existing in ASF is used)
BERKELEY_TO_CMU = [
    # Base / pelvis
    ("base", ["root"]),
    # Left leg
    ("leg_left_hip_roll", ["lhipjoint"]),
    ("leg_left_hip_pitch", ["lhipjoint", "lupperleg", "lfemur"]),
    ("leg_left_knee_pitch", ["lknee"]),
    ("leg_left_ankle_pitch", ["llowerleg", "ltibia", "lfoot"]),
    ("leg_left_ankle_roll", ["lfoot"]),
    # Right leg
    ("leg_right_hip_roll", ["rhipjoint"]),
    ("leg_right_hip_pitch", ["rhipjoint", "rupperleg", "rfemur"]),
    ("leg_right_knee_pitch", ["rknee"]),
    ("leg_right_ankle_pitch", ["rlowerleg", "rtibia", "rfoot"]),
    ("leg_right_ankle_roll", ["rfoot"]),
    # Left arm
    ("arm_left_shoulder_pitch", ["lclavicle", "lupperarm"]),
    ("arm_left_shoulder_roll", ["lupperarm"]),
    ("arm_left_elbow_pitch", ["lelbow"]),
    ("arm_left_elbow_roll", ["llowerarm"]),
    ("arm_left_hand_link", ["lhand"]),
    # Right arm (hand frame may be arm_right_hand_link or arm_hand_r)
    ("arm_right_shoulder_pitch", ["rclavicle", "rupperarm"]),
    ("arm_right_shoulder_roll", ["rupperarm"]),
    ("arm_right_elbow_pitch", ["relbow"]),
    ("arm_right_elbow_roll", ["rlowerarm"]),
    ("arm_right_hand_link", ["rhand"]),
]

# Optional: frames that are not in every URDF (e.g. arm_hand_r as alias for right hand)
BERKELEY_TO_CMU_ALT = [
    ("arm_hand_r", ["rhand"]),
    ("arm_left_hand_l", ["lhand"]),
]
