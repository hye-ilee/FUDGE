######################################################################################
# command.py
# This module provides functionality to manage motion commands in a Streamlit application.
# It allows users to add, delete, and apply motion commands to motion data,
# and to render modified motion videos based on user inputs.
######################################################################################

from config import MOTION_COMMANDS
import streamlit as st
from sentence_transformers import util
from natsort import natsorted
import os
import re
import numpy as np
import subprocess
from utils.motion_modifier import apply_modifications_to_motion
from config import (
    MOTION_DIR, MUSIC_DIR, MODIFIED_VIDEO_DIR
)
from utils.data_loader import motion_data_load_process, find_music_and_time
from utils.video import render_modified_motion

def init_command_matrix():
    if "command_matrix" not in st.session_state:
        st.session_state.command_matrix = [[], [], [], []]  # 4 blocks

def encode_commands(model):
    return model.encode(MOTION_COMMANDS, convert_to_tensor=True)


def find_best_command(user_input, model, command_embeddings):
    user_embedding = model.encode(user_input, convert_to_tensor=True)
    best_idx = util.cos_sim(user_embedding, command_embeddings).argmax().item()
    return MOTION_COMMANDS[best_idx]


def add_command_to_session(joint, delta, frame_idx, matched_command):
    idx = frame_idx // 256
    cmd = [joint, delta, frame_idx, matched_command]
    st.session_state.command_matrix[idx].append(cmd)


def clear_command_matrix():
    st.session_state.command_matrix = [[], [], [], []]
    st.rerun()
                
def delete_command(block_idx, cmd_idx):
    st.session_state.command_matrix[block_idx].pop(cmd_idx)
    st.rerun()
    
def apply_commands_and_render_videos(command_array, npy_title):
    npy_core = npy_title.replace('.npy', '')

    filtered_files = [
        f for f in natsorted(os.listdir(MOTION_DIR))
        if re.search(r'\d{3}_[A-Za-z]+', f) and re.search(r'\d{3}_[A-Za-z]+', f).group() == npy_core
    ]

    for motion_file, commands in zip(filtered_files, command_array):
        print("[PROCESS]", motion_file)
        motion_path = os.path.join(MOTION_DIR, motion_file)

        music_file, start_sec, end_sec = find_music_and_time(motion_file, MUSIC_DIR)
        if music_file is None:
            print(f"[SKIP] Cannot find music for {motion_file}")
            continue

        trimmed_music = os.path.join(MODIFIED_VIDEO_DIR, f"{motion_file[:-4]}_trimmed.wav")
        subprocess.run([
            'ffmpeg', '-y',
            '-i', music_file,
            '-ss', str(start_sec),
            '-to', str(end_sec),
            '-c', 'copy',
            trimmed_music
        ])

        original_data = motion_data_load_process(motion_path)
        original_name = os.path.splitext(os.path.basename(motion_path))[0]
        np.save(os.path.join(MODIFIED_VIDEO_DIR, original_name + "_original.npy"), original_data)

        modified_data = apply_modifications_to_motion(original_data, commands)

        render_modified_motion(
            music_file=trimmed_music,
            modified_data=modified_data,
            original_name=original_name
        )

    print("done")
    