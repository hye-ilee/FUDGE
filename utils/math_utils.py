######################################################################################
# math.py
# This module provides utility functions for mathematical operations related to 3D transformations,
# including conversions between different representations of rotations (quaternions, axis-angle, 6D rotation vectors),
# and camera pose calculations.
# It also includes a function to compute a camera pose from eye, center, and up vectors.
######################################################################################

from pytorch3d.transforms import (
    axis_angle_to_matrix, matrix_to_axis_angle,
    matrix_to_quaternion, matrix_to_rotation_6d,
    quaternion_to_matrix, rotation_6d_to_matrix
)
import numpy as np

def quat_to_6v(q):
    assert q.shape[-1] == 4
    return matrix_to_rotation_6d(quaternion_to_matrix(q))

def quat_from_6v(q):
    assert q.shape[-1] == 6
    return matrix_to_quaternion(rotation_6d_to_matrix(q))

def ax_to_6v(q):
    assert q.shape[-1] == 3
    return matrix_to_rotation_6d(axis_angle_to_matrix(q))

def ax_from_6v(q):
    assert q.shape[-1] == 6
    return matrix_to_axis_angle(rotation_6d_to_matrix(q))

def look_at(eye, center, up):
    front = eye - center
    front = front / np.linalg.norm(front)
    right = np.cross(up, front)
    right = right / np.linalg.norm(right)
    up_new = np.cross(front, right)
    camera_pose = np.eye(4)
    camera_pose[:3, :3] = np.stack([right, up_new, front]).T
    camera_pose[:3, 3] = eye
    return camera_pose
