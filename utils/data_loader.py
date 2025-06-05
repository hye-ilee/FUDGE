######################################################################################
# data_loader.py
# This module provides functions to load and process motion data from files,
# find associated music files, and manage video paths. 
######################################################################################

import pickle
import os
import numpy as np
import torch
import re
from utils.math_utils import ax_from_6v
from config import FPS, SEGMENT_LEN, MODIFIED_VIDEO_DIR, ORIGINAL_VIDEO_DIR
import shutil
from datetime import datetime

def motion_data_load_process(motionfile):
    ext = motionfile.split(".")[-1]
    if ext == "pkl":
        pkl_data = pickle.load(open(motionfile, "rb"))
        smpl_poses = pkl_data["smpl_poses"]
        modata = np.concatenate((pkl_data["smpl_trans"], smpl_poses), axis=1)
        if modata.shape[1] == 69:
            hand_zeros = np.zeros([modata.shape[0], 90], dtype=np.float32)
            modata = np.concatenate((modata, hand_zeros), axis=1)
        assert modata.shape[1] == 159
        modata[:, 1] += 0
        return modata
    elif ext == "npy":
        modata = np.load(motionfile)
        if len(modata.shape) == 3 and modata.shape[1] % 8 == 0:
            print("modata has 3 dim , reshape the batch to time!!!")
            modata = modata.reshape(-1, modata.shape[-1])
        if modata.shape[-1] in [315, 319, 135, 139]:
            if modata.shape[-1] in [315, 319]:
                if modata.shape[-1] == 319:
                    modata = modata[:, 4:]
                rot6d = torch.from_numpy(modata[:, 3:])
                T, C = rot6d.shape
                axis = ax_from_6v(rot6d.reshape(-1, 6)).view(T, -1).cpu().numpy()
                modata = np.concatenate((modata[:, :3], axis), axis=1)
            elif modata.shape[-1] in [135, 139]:
                if modata.shape[-1] == 139:
                    modata = modata[:, 4:]
                rot6d = torch.from_numpy(modata[:, 3:])
                T, C = rot6d.shape
                axis = ax_from_6v(rot6d.reshape(-1, 6)).view(T, -1).cpu().numpy()
                hand_zeros = np.zeros([T, 90], dtype=np.float32)
                modata = np.concatenate((modata[:, :3], axis, hand_zeros), axis=1)
        elif modata.shape[-1] == 159:
            pass
        else:
            raise ValueError("shape error!")
        modata[:, 1] += 0
        return modata
    else:
        raise ValueError("Unsupported file extension for motion data.")
    
def find_music_and_time(file_name, music_dir):
    pattern = re.compile(r"g(\d{3})g_l(\d{3})")
    match = pattern.search(file_name)
    if not match:
        return None, None, None

    g_idx = int(match.group(1))
    l_idx = int(match.group(2))
    total_idx = g_idx * 4 + l_idx

    parts = file_name.split("_")
    if len(parts) < 3:
        return None, None, None

    music_id = parts[2]
    title = parts[3]
    keyword = f"{music_id}_{title}"

    music_file = next((
        os.path.join(music_dir, f)
        for f in os.listdir(music_dir)
        if keyword in f and f.endswith(".wav")
    ), None)

    if music_file is None:
        return None, None, None

    seconds_per_segment = SEGMENT_LEN / FPS
    start_sec = round(total_idx * seconds_per_segment, 2)
    end_sec = round(start_sec + seconds_per_segment, 2)

    return music_file, start_sec, end_sec

def sanitize_title(title: str) -> str:
    """Remove spaces and special chars from video title for filename."""
    return title.replace(" ", "").replace("!", "").replace("?", "")


def get_npy_title(video_index: int, video_title: str) -> str:
    sanitized_title = sanitize_title(video_title)
    return f"{video_index:03d}_{sanitized_title}.npy"


def get_original_video_path(video_index: int, video_title: str) -> str:
    sanitized_title = sanitize_title(video_title)
    return os.path.join(ORIGINAL_VIDEO_DIR, f"{video_index:03d}_{sanitized_title}_merged.mp4")

def clean_modified_videos():
    for item in os.listdir(MODIFIED_VIDEO_DIR):
        item_path = os.path.join(MODIFIED_VIDEO_DIR, item)
        if item == "history":
            continue
        if item == "merged":
            move_merged_to_history()
            continue
        if os.path.isfile(item_path):
            os.remove(item_path)
        elif os.path.isdir(item_path):
            shutil.rmtree(item_path)

def move_merged_to_history():
    history_dir = os.path.join(MODIFIED_VIDEO_DIR, "history")
    os.makedirs(history_dir, exist_ok=True)
    merged_dir = os.path.join(MODIFIED_VIDEO_DIR, "merged")
    if not os.path.exists(merged_dir):
        return

    for f in os.listdir(merged_dir):
        fp = os.path.join(merged_dir, f)
        if os.path.isfile(fp):
            base, ext = os.path.splitext(f)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            new_fname = f"{base}_{timestamp}{ext}"
            dst_path = os.path.join(history_dir, new_fname)
            shutil.move(fp, dst_path)
        elif os.path.isdir(fp):
            shutil.rmtree(fp)
