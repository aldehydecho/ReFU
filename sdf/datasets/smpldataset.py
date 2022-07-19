import torch
import torch.utils.data as data
import numpy as np
import os

import pickle
import trimesh
from tqdm import tqdm, trange
import random

import sys

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

import igrutils.general as utils

class SMPLDataSet(data.Dataset):

    def __init__(self, dataset_path, file_name_list_file, points_batch=8000, with_normals=False, return_index=False):

        self.dataset_path = dataset_path
        
        with open(file_name_list_file, 'rb') as inputfile:
            self.file_name_list = pickle.load(inputfile)

        self.points_batch = points_batch

        self.with_normals = with_normals

        self.return_index = return_index

    def load_data(self, index):
        info_str, id = self.file_name_list[index]

        shape_str, style_str, seq_str = info_str.split('_')

        beta = torch.from_numpy(np.load(os.path.join(self.dataset_path, 'shape/beta_{}.npy'.format(shape_str)))[0:10]).float()
        theta = torch.from_numpy(np.load(os.path.join(self.dataset_path, 'pose/{}_{}/poses_{}.npz'.format(shape_str, style_str, seq_str)))['thetas'][id, :]).float()

        obj_dir_path = os.path.join(self.dataset_path, 'pose_obj_total', info_str, 'body')

        cloth_obj_dir_path = os.path.join(self.dataset_path, 'pose_obj_total', info_str, 'garment')

        mnlfold = np.load(os.path.join(obj_dir_path, 'manifold_'+str(id)+'.npy')).astype(np.float32)

        point_set_mnlfold = torch.from_numpy(mnlfold[:, :3])

        sdf_pair = np.load(os.path.join(obj_dir_path, 'sdf_'+str(id)+'.npy')).astype(np.float32)

        cloth_sdf_pair = np.load(os.path.join(cloth_obj_dir_path, 'garment_sdf_'+str(id)+'.npy')).astype(np.float32)

        sdf_pair = np.concatenate((sdf_pair, cloth_sdf_pair), axis=0)

        sdf_point = torch.from_numpy(sdf_pair[:, 0:3])

        sdf_value = torch.from_numpy(sdf_pair[:, 3])

        if self.with_normals == True:
            normals = torch.from_numpy(mnlfold[:, -3:])

            return beta, theta, point_set_mnlfold, normals, sdf_point, sdf_value

        else:
            return beta, theta, point_set_mnlfold, sdf_point, sdf_value


    def __getitem__(self, index):

        if self.with_normals == True:
            beta, theta, point_set_mnlfold, normals, sdf_point, sdf_value = self.load_data(index)
        else:
            beta, theta, point_set_mnlfold, sdf_point, sdf_value = self.load_data(index)
            normals = torch.empty(0)

        random_idx = torch.randperm(point_set_mnlfold.shape[0])[:self.points_batch]

        point_set_mnlfold = torch.index_select(point_set_mnlfold, 0, random_idx)

        if self.with_normals:
            normals = torch.index_select(normals, 0, random_idx)

        random_idx = torch.randperm(sdf_point.shape[0])[:(self.points_batch+self.points_batch//8)]

        sdf_point = torch.index_select(sdf_point, 0, random_idx)
        sdf_value = torch.index_select(sdf_value, 0, random_idx)

        if self.return_index == True:
            return beta, theta, point_set_mnlfold, normals, sdf_point, sdf_value, self.file_name_list[index][0] + '_'+ str(self.file_name_list[index][1]), index
        else:
            return beta, theta, point_set_mnlfold, normals, sdf_point, sdf_value, self.file_name_list[index][0] + '_'+ str(self.file_name_list[index][1])

    def __len__(self):
        return len(self.file_name_list)



    def __init__(self, dataset_path, file_name_list_file, points_batch=8000, with_normals=False):

        self.dataset_path = dataset_path
        
        with open(file_name_list_file, 'rb') as inputfile:
            self.file_name_list = pickle.load(inputfile)

        self.points_batch = points_batch

        self.with_normals = with_normals

    def load_data(self, index):
        info_str, id = self.file_name_list[index]

        shape_str, style_str, seq_str = info_str.split('_')

        beta = torch.from_numpy(np.load(os.path.join(self.dataset_path, 'shape/beta_{}.npy'.format(shape_str)))[0:10]).float()
        theta = torch.from_numpy(np.load(os.path.join(self.dataset_path, 'pose/{}_{}/poses_{}.npz'.format(shape_str, style_str, seq_str)))['thetas'][id, :]).float()

        obj_dir_path = os.path.join(self.dataset_path, 'pose_obj_total', info_str, 'body')

        vertices = torch.from_numpy(np.array(trimesh.load(os.path.join(obj_dir_path, str(id)+'.obj')).vertices)).float()

        if self.with_normals == True:
            with open(os.path.join(obj_dir_path, 'vn_'+str(id)+'.pkl'), 'rb') as normal_file:
                normals = torch.from_numpy(pickle.load(normal_file)).float()
            
            return beta, theta, vertices, normals

        else:
            return beta, theta, vertices


    def __getitem__(self, index):

        if self.with_normals == True:
            beta, theta, vertices, normals = self.load_data(index)
        else:
            beta, theta, vertices = self.load_data(index)
            normals = torch.empty(0)

        if self.points_batch is not None:

            random_idx = torch.randperm(vertices.shape[0])[:self.points_batch]

            vertices = torch.index_select(vertices, 0, random_idx)

            if self.with_normals:
                normals = torch.index_select(normals, 0, random_idx)

        return beta, theta, vertices, normals, self.file_name_list[index][0] + '_'+ str(self.file_name_list[index][1])

    def __len__(self):
        return len(self.file_name_list)