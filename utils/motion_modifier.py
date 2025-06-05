######################################################################################
# motion_modifier.py
# This module modifies motion data by applying specified changes to joint rotations
# based on user commands. It supports pre-blending and post-blending of changes
# to ensure smooth transitions in the motion data.
######################################################################################

import os
import numpy as np
os.environ["PYOPENGL_PLATFORM"] = "egl"
from config import FPS, SMPL_JOINT_NAMES

def modify_motion_frame(data, frame_idx, joint_name, delta):
    joint_idx = SMPL_JOINT_NAMES.index(joint_name)
    rot_start = 3 + 3 * joint_idx
    rot_end = rot_start + 3
    modified = data.copy()
    if isinstance(delta, (float, int)):
        modified[frame_idx, rot_start] += delta
    elif isinstance(delta, (np.ndarray, list)) and len(delta) == 3:
        modified[frame_idx, rot_start:rot_end] += np.array(delta)
    else:
        raise ValueError("delta must be scalar or 3D vector")
    return modified


def apply_modifications_to_motion(original_data,
    command_array=None,):
    print(command_array)

    modified_data = original_data.copy()
    blend_margin = int(0.5 * FPS)

    for i in range (len(command_array)):
        joint_name, delta, frame_idx, ai_command = command_array[i]
        frame_idx = frame_idx % 256
        # Pre-Blend
        for i, f in enumerate(range(max(0, frame_idx - blend_margin), frame_idx)):
            ratio = (i + 1) / blend_margin
            delta_scaled = np.array(delta) * ratio
            modified_data = modify_motion_frame(modified_data, f, joint_name, delta_scaled)
        # Main motion
        T = original_data.shape[0]
        if 0 <= frame_idx < T:
            modified_data = modify_motion_frame(modified_data, frame_idx, joint_name, delta)
        # Post-Blend
        for i, f in enumerate(range(frame_idx + 1, min(T, frame_idx + 1 + blend_margin))):
            ratio = 1 - (i + 1) / blend_margin
            delta_scaled = np.array(delta) * ratio
            modified_data = modify_motion_frame(modified_data, f, joint_name, delta_scaled)

    return modified_data

