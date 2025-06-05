######################################################################################
# FUDGE: Freeform User Dance Generator and Editor
# This application allows users to input natural language commands to modify dance videos.
# It uses a pre-trained model to interpret commands and applies them to dance motion data.
######################################################################################

import streamlit as st
import math
from sentence_transformers import SentenceTransformer
from config import (
    VIDEO_CHOICES, SMPL_JOINT_NAMES,
    ROTATION_AXES, ROTATION_COMMANDS
)
from utils.command import (
    init_command_matrix, encode_commands, find_best_command,
    add_command_to_session, clear_command_matrix, delete_command, apply_commands_and_render_videos
)
from utils.video import (
    merge_videos, find_modified_video_path, side_by_side_video,
)
from utils.data_loader import (
    get_npy_title, get_original_video_path, clean_modified_videos
)

def FUDGE():
    st.set_page_config(layout="wide")
    st.title("🕺 FUDGE Demo")
    
    init_command_matrix()
    clean_modified_videos()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🎥 Original Video")   
        # Video selection
        selected = st.selectbox(
            "Select a video:",
            options=list(VIDEO_CHOICES.keys()),
            format_func=lambda x: f"{x:03d} - {VIDEO_CHOICES[x][0]} ({VIDEO_CHOICES[x][1]})"
        )
        video_index = selected
        video_title = VIDEO_CHOICES[video_index][0]
        npy_title = get_npy_title(video_index, video_title)
        original_video_path = get_original_video_path(video_index, video_title)
    
        
        # Show original video
        st.video(original_video_path)

    with col2:
        # Semantic Command Section
        st.subheader("🧠 Semantic Command")

        model = SentenceTransformer('all-MiniLM-L6-v2')
        command_embeddings = encode_commands(model)

        user_input = st.text_input("Enter a natural language command:", "")

        if user_input.strip():
            matched_command = find_best_command(user_input, model, command_embeddings)
            st.markdown(f"✅ **Interpreted:** `{matched_command}`")

            joint_idx, axis_idx = ROTATION_COMMANDS[matched_command]
            joint = SMPL_JOINT_NAMES[joint_idx]
            axis = ROTATION_AXES[axis_idx]

            angle = st.slider("Rotation angle (degrees)", min_value=0, max_value=180, value=30)
            angle_rad = angle * 3.1416 / 180
            delta = [a * angle_rad for a in axis]

            sec = st.number_input("Target time (in seconds)", min_value=0.0, max_value=34.0, value=5.0, step=0.1)
            frame_idx = int(sec * 30)

            spacer, col_add = st.columns([3, 1])
            with col_add:
                if st.button("➕ Add Command"):
                    add_command_to_session(joint, delta, frame_idx, matched_command)

        # Show queued commands
        st.markdown("---")
        st.subheader("📋 Queued Commands")

        for block_idx, block in enumerate(st.session_state.command_matrix):
            if len(block) > 0:
                for cmd_idx, cmd in enumerate(block):
                    if len(cmd) == 4:
                        joint, delta, frame, matched_command = cmd
                        col1, col2 = st.columns([9, 1])
                        with col1:
                            st.write(f" - `{matched_command}` `{math.sqrt(sum(d**2 for d in delta))* 180 // math.pi}` degrees at `{frame/30:.1f}` sec")
                        with col2:
                            if st.button("🗑️", key=f"delete_{block_idx}_{cmd_idx}"):
                                delete_command(block_idx, cmd_idx)
                    else:
                        st.write(" - Invalid command format")

        spacer, col_clear, col_run = st.columns([5.5,2,3])
        with col_clear:
            if st.button("🗑️ Clear All"):
                clear_command_matrix()
         
                
        with col_run:
            if st.button("🚀 Run All Commands"):
                # Run all commands and render 4 separate videos
                clean_modified_videos()
                apply_commands_and_render_videos(st.session_state.command_matrix, npy_title)

                merged_dir = merge_videos()
        modified_video_path = find_modified_video_path(npy_title)

    # Show side-by-side comparison video if available
    if modified_video_path:
        comparison_path = side_by_side_video(original_video_path, modified_video_path)
        if comparison_path:
            st.subheader("🆚 Side-by-side Comparison")
            st.video(comparison_path)


if __name__ == "__main__":
    FUDGE()
