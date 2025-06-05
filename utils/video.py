######################################################################################
# video.py
# This module provides functionality to merge videos, render modified motion videos, 
# and create side-by-side comparisons of original and modified videos.
######################################################################################

from collections import defaultdict
import os
from natsort import natsorted
import subprocess
from config import MODIFIED_VIDEO_DIR
import streamlit as st
import re
import numpy as np
import torch
import cv2
import pyrender
import trimesh
from tqdm import tqdm
from smplx import SMPLX
from config import (
    MODIFIED_VIDEO_DIR, FPS
)
from utils.math_utils import look_at



def merge_videos():
    output_dir = os.path.join(MODIFIED_VIDEO_DIR, "merged")
    os.makedirs(output_dir, exist_ok=True)

    pattern = re.compile(r".*dod_(\d+)_(\d+)_([^_]+)_test_.*z\.mp4")
    title_groups = defaultdict(list)

    for fname in os.listdir(MODIFIED_VIDEO_DIR):
        if not fname.endswith(".mp4") or "z.mp4" not in fname:
            continue
        match = pattern.match(fname)
        if match:
            dod_idx, song_id, title = match.groups()
            key = f"{song_id}_{title}"
            title_groups[key].append((dod_idx, fname))

    for key, files in title_groups.items():
        files = natsorted(files, key=lambda x: x[0])
        input_txt_path = os.path.join(MODIFIED_VIDEO_DIR, f"inputs_{key}.txt")

        with open(input_txt_path, "w") as f:
            for _, fname in files:
                full_path = os.path.abspath(os.path.join(MODIFIED_VIDEO_DIR, fname))
                f.write(f"file '{full_path}'\n")

        output_path = os.path.join(output_dir, f"{key}_merged.mp4")

        result = subprocess.run([
            "ffmpeg", "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", input_txt_path,
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-crf", "23",
            "-c:a", "aac",
            output_path
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        os.remove(input_txt_path)

        if result.returncode != 0:
            st.error(f"⛔ Failed to merge {key}: {result.stderr.decode()}")
        else:
            st.info(f"✔ Merged: {output_path}")

    return output_dir


def find_modified_video_path(npy_title):
    merged_dir = os.path.join(MODIFIED_VIDEO_DIR, "merged")
    for f in os.listdir(merged_dir):
        if f.endswith(".mp4") and npy_title[:-4] in f:
            return os.path.join(merged_dir, f)
    return None


def side_by_side_video(original_path, modified_path):
    merged_video_path = f"/tmp/compare_{os.path.basename(original_path)[:-4]}_vs_modified.mp4"
    try:
        subprocess.run([
            "ffmpeg", "-y",
            "-i", original_path,
            "-i", modified_path,
            "-filter_complex", "hstack=inputs=2",
            merged_video_path
        ], check=True)
    except subprocess.CalledProcessError as e:
        st.error(f"⚠️ Failed to merge videos: {e}")
        return None
    return merged_video_path


def render_modified_motion(
    music_file=None,
    modified_data=None,
    original_name=None,):
    
    modified_name = f"modified__{original_name}"
    np.save(os.path.join(MODIFIED_VIDEO_DIR, modified_name + ".npy"), modified_data)

    visualizer = MovieMaker(save_path=MODIFIED_VIDEO_DIR)
    visualizer.run(modified_data, tab=modified_name, music_file=music_file)


class MovieMaker():
    def __init__(self, save_path, device="0") -> None:
        
        self.mag = 2
        self.eyes = np.array([[3,-3,2], [0,0,-2], [0,0,4], [-8,-8,1], [0,-2,4], [0,2,4]])
        self.centers = np.array([[0,0,0],[0,0,0],[0,0.5,0],[0,0,-1], [0,0.5,0], [0,0.5,0]])
        self.ups = np.array([[0,0,1],[0,1,0],[0,1,0],[0,0,-1], [0,1,0], [0,1,0]])
        self.save_path = save_path
        
        self.img_size = (1200,1200)

    
        SMPLX_path = "/root/CS470/LODGE/data/human/datasets/smpl_model/smplx/SMPLX_NEUTRAL.npz"
        trimesh_path = '/root/CS470/LODGE/data/NORMAL_new.obj'

        self.smplx = SMPLX(SMPLX_path, use_pca=False, flat_hand_mean=True).eval()
        self.smplx.to(f'cuda:{0}').eval()

        self.scene = pyrender.Scene()
        camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
        camera_pose = look_at(self.eyes[5], self.centers[5], self.ups[5])       # 2
        self.scene.add(camera, pose=camera_pose)
        light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
        self.scene.add(light, pose=camera_pose)
        self.r = pyrender.OffscreenRenderer(self.img_size[0], self.img_size[1])
        
        self.mesh = trimesh.load(trimesh_path)
        floor_mesh  = pyrender.Mesh.from_trimesh(self.mesh)   
        floor_node = self.scene.add(floor_mesh)


    def save_video(self, save_path, color_list):
        f = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        videowriter = cv2.VideoWriter(save_path,f,FPS,self.img_size)
        for i in range(len(color_list)):
            videowriter.write(color_list[i][:,:,::-1])
        videowriter.release()

    def get_imgs(self, motion):
        meshes = self.motion2mesh(motion)
        imgs = self.render_imgs(meshes)
        return np.concatenate(imgs, axis=1)

    def motion2mesh(self, motion):
        output = self.smplx.forward(
            betas = torch.zeros([motion.shape[0], 10]).to(motion.device),
            transl = motion[:,:3],
            global_orient = motion[:,3:6],
            body_pose = motion[:,6:69],
            jaw_pose = torch.zeros([motion.shape[0], 3]).to(motion),
            leye_pose = torch.zeros([motion.shape[0], 3]).to(motion),
            reye_pose = torch.zeros([motion.shape[0], 3]).to(motion),
            left_hand_pose = torch.zeros([motion.shape[0], 45]).to(motion),
            right_hand_pose = torch.zeros([motion.shape[0], 45]).to(motion),
            expression= torch.zeros([motion.shape[0], 10]).to(motion),
        )

        
        meshes = []
        for i in range(output.vertices.shape[0]):
            mesh = trimesh.Trimesh(output.vertices[i].cpu(), self.smplx.faces)
            meshes.append(mesh)
        
        return meshes


    def render_multi_view(self, meshes, music_file, tab='', eyes=None, centers=None, ups=None, views=1):
        if eyes and centers and ups:
            assert eyes.shape == centers.shape == ups.shape
        else:
            eyes = self.eyes
            centers = self.centers
            ups = self.ups
        
        for i in range(views):
            color_list = self.render_single_view(meshes, eyes[1], centers[1], ups[1])
            movie_file = os.path.join(self.save_path, tab + '-' + str(i) + '.mp4')
            output_file = os.path.join(self.save_path, tab + '-' + str(i) + '-music.mp4')
            self.save_video(movie_file, color_list)
            if music_file is not None:
                subprocess.run(['ffmpeg','-i',movie_file,'-i',music_file,'-shortest',output_file])
            else:
                subprocess.run(['ffmpeg','-i',movie_file,output_file])
                os.remove(movie_file)

            
            

    def render_single_view(self, meshes):
        num = len(meshes)
        color_list = []
        for i in tqdm(range(num)):
            mesh_nodes = []
            for mesh in meshes[i]:
                render_mesh = pyrender.Mesh.from_trimesh(mesh)   
                mesh_node = self.scene.add(render_mesh)
                mesh_nodes.append(mesh_node)
            color, _ = self.r.render(self.scene, flags=pyrender.RenderFlags.SHADOWS_DIRECTIONAL)
            color = color.copy()
            color_list.append(color)
            for mesh_node in mesh_nodes:
                self.scene.remove_node(mesh_node)
        return color_list
    
    def render_imgs(self, meshes):
        colors = []
        for mesh in meshes:
            render_mesh = pyrender.Mesh.from_trimesh(mesh)   
            mesh_node = self.scene.add(render_mesh)
            color, _ = self.r.render(self.scene, flags=pyrender.RenderFlags.SHADOWS_DIRECTIONAL)
            colors.append(color)
            self.scene.remove_node(mesh_node)


        return colors
    
    def run(self, seq_rot, music_file=None, tab='', save_pt=False):
        if isinstance(seq_rot, np.ndarray):
            seq_rot = torch.tensor(seq_rot, dtype=torch.float32, device=f'cuda:{0}')

        if save_pt:
            torch.save(seq_rot.detach().cpu(), os.path.join(self.save_path, tab +'_pose.pt'))

        B, D = seq_rot.shape
        output = self.smplx.forward(
            betas = torch.zeros([seq_rot.shape[0], 10]).to(seq_rot.device),
            transl = seq_rot[:,:3],
            global_orient = seq_rot[:,3:6],
            body_pose = seq_rot[:,6:69],
            jaw_pose = torch.zeros([seq_rot.shape[0], 3]).to(seq_rot),
            leye_pose = torch.zeros([seq_rot.shape[0], 3]).to(seq_rot),
            reye_pose = torch.zeros([seq_rot.shape[0], 3]).to(seq_rot),
            left_hand_pose = torch.zeros([seq_rot.shape[0], 45]).to(seq_rot),
            right_hand_pose = torch.zeros([seq_rot.shape[0], 45]).to(seq_rot),
            expression= torch.zeros([seq_rot.shape[0], 10]).to(seq_rot),
            )
        
        N, V, DD = output.vertices.shape                # 150, 6890, 3
        vertices = output.vertices.reshape((B, -1, V, DD))  #  # 150, 1, 6890, 3
        
        meshes = []
        for i in range(B):
            view = []
            for v in vertices[i]:
                mesh = trimesh.Trimesh(output.vertices[i].cpu(), self.smplx.faces)
                view.append(mesh)
            meshes.append(view)

        color_list = self.render_single_view(meshes)
        movie_file = os.path.join(self.save_path, tab + 'tmp.mp4')
        output_file = os.path.join(self.save_path, tab + 'z.mp4')
        self.save_video(movie_file, color_list)
        if music_file is not None:
            subprocess.run(['/root/miniconda3/envs/lodge/bin/ffmpeg','-i',movie_file,'-i',music_file,'-shortest',output_file])
        else:
            subprocess.run(['/root/miniconda3/envs/lodge/bin/ffmpeg', '-i', movie_file, output_file])
        os.remove(movie_file)
        
