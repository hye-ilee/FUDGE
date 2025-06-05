import numpy as np
import torch
from pytorch3d.transforms import rotation_6d_to_matrix, matrix_to_axis_angle

def convert_159_to_139_fixed(data_159):
    trans = data_159[:, :3]
    rot6d = data_159[:, 3:135]
    others = data_159[:, 135:205]  # 70-dim extra
    rot6d_tensor = torch.tensor(rot6d.reshape(-1, 6), dtype=torch.float32)
    axis = matrix_to_axis_angle(rotation_6d_to_matrix(rot6d_tensor)).view(data_159.shape[0], -1).numpy()
    return np.concatenate([trans, axis, others], axis=1)

paths = [
    "modified__dod_0_001_PinkVenom_test_001_mMH1g000g_l000.npy",
    "modified__dod_1_001_PinkVenom_test_001_mMH1g000g_l001.npy",
    "modified__dod_2_001_PinkVenom_test_001_mMH1g000g_l002.npy",
    "modified__dod_3_001_PinkVenom_test_001_mMH1g000g_l003.npy",
]

merged = []
for path in paths:
    data = np.load(path)
    merged.append(convert_159_to_139_fixed(data))
merged_data = np.concatenate(merged, axis=0)
np.save("merged_modified_001_PinkVenom_139.npy", merged_data)
