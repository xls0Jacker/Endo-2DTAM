import glob
import os
from pathlib import Path
from typing import Dict, List, Optional, Union
import numpy as np
import torch
from natsort import natsorted
import cv2
from scipy.spatial.transform import Rotation as R
from .basedataset import GradSLAMDataset


class SimCol3DDataset(GradSLAMDataset):
    def __init__(
        self,        
        config_dict,
        basedir,
        sequence,
        stride: Optional[int] = None,
        start: Optional[int] = 0,
        end: Optional[int] = -1,
        desired_height: Optional[int] = 540,
        desired_width: Optional[int] = 675,
        load_embeddings: Optional[bool] = False,
        embedding_dir: Optional[str] = "embeddings",
        embedding_dim: Optional[int] = 512,
        train_or_test: Optional[str] = 'all',
        **kwargs,
    ):
        self.input_folder = os.path.join(basedir, sequence)
        self.mode = train_or_test
        self.use_dep = config_dict['use_dep']
        super().__init__(
            config_dict,
            stride=stride,
            start=start,
            end=end,
            desired_height=desired_height,
            desired_width=desired_width,
            load_embeddings=load_embeddings,
            embedding_dir=embedding_dir,
            embedding_dim=embedding_dim,
            **kwargs,
        )
        
    def train_test_split(self, stride):
        # stride 1
        all_idx = set(range(self.end))
        train_idx = None
        eval_idx = set(range(self.start + 7, self.end, 8)) # we don't expect eval in early frames
        train_idx = all_idx - eval_idx
        eval_idx = sorted(list(eval_idx))
        train_idx = sorted(list(train_idx))
            
        if self.mode == 'test':
            self.color_paths = [self.color_paths[i] for i in eval_idx]
            self.depth_paths = [self.depth_paths[i] for i in eval_idx]
            if self.load_embeddings:
                self.embedding_paths = [self.embedding_paths[i] for i in eval_idx]
            self.poses = [self.poses[i] for i in eval_idx]
            self.retained_inds = torch.arange(self.num_imgs)[eval_idx]
        elif self.mode == 'train':
            self.color_paths = [self.color_paths[i] for i in train_idx]
            self.depth_paths = [self.depth_paths[i] for i in train_idx]
            if self.load_embeddings:
                self.embedding_paths = [self.embedding_paths[i] for i in train_idx]
            self.poses = [self.poses[i] for i in train_idx]
            self.retained_inds = torch.arange(self.num_imgs)[train_idx]
        
            self.color_paths = self.color_paths[self.start : len(self.color_paths) : stride]
            self.depth_paths = self.depth_paths[self.start : len(self.depth_paths) : stride]
            if self.load_embeddings:
                self.embedding_paths = self.embedding_paths[self.start : len(self.embedding_paths) : stride]
            self.poses = self.poses[self.start : len(self.poses) : stride]
            # Tensor of retained indices (indices of frames and poses that were retained)
            self.retained_inds = torch.arange(self.num_imgs)[self.start : len(self.retained_inds) : stride]
        else:
            super().train_test_split(stride)

    def get_filepaths(self):
        color_paths = natsorted(glob.glob(f"{self.input_folder}/FrameBuffer_*.png"))[::3]
        depth_paths = natsorted(glob.glob(f"{self.input_folder}/Depth_*.png"))[::3]
        embedding_paths = None
        if self.load_embeddings:
            embedding_paths = natsorted(glob.glob(f"{self.input_folder}/{self.embedding_dir}/*.pt"))
        return color_paths, depth_paths, embedding_paths
    
    def load_poses(self):
        """
        :param scene: Index of trajectory
        :param root: Root folder of dataset
        :return: all camera poses as quaternion vector and 4x4 projection matrix
        """
        main_dir = self.input_folder.split('Frames_')[0]
        scene = self.input_folder.split('Frames_')[1]
        
        locations = []
        rotations = []
        loc_reader = open(main_dir+ 'SavedPosition_' + scene + '.txt', 'r')
        rot_reader = open(main_dir + 'SavedRotationQuaternion_' + scene + '.txt', 'r')
        for line in loc_reader:
            locations.append(list(map(float, line.split())))

        for line in rot_reader:
            rotations.append(list(map(float, line.split())))

        locations = np.array(locations[::3])
        rotations = np.array(rotations[::3])
        

        # r = R.from_quat(rotations).as_dcm()
        r = R.from_quat(rotations).as_matrix()

        TM = np.eye(4)
        TM[1, 1] = -1

        poses_mat = []
        for i in range(locations.shape[0]):
            ri = r[i]
            Pi = np.concatenate((ri, locations[i].reshape((3, 1))), 1)
            Pi = np.concatenate((Pi, np.array([0.0, 0.0, 0.0, 1.0]).reshape((1, 4))), 0)
            Pi_left = TM @ Pi @ TM   # Translate between left and right handed systems
            poses_mat.append(torch.Tensor(Pi_left).float())
        return poses_mat

    def read_embedding_from_file(self, embedding_file_path):
        embedding = torch.load(embedding_file_path)
        return embedding.permute(0, 2, 3, 1)  # (1, H, W, embedding_dim)
