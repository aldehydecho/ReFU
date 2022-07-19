import torch
import os
import sys
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import pickle
from models import ops

import time

from torch.autograd import grad

from psbody.mesh import Mesh
from psbody.mesh.geometry.vert_normals import VertNormals
from psbody.mesh.geometry.tri_normals import TriNormals
from psbody.mesh.search import AabbTree

def get_nearest_points_and_normals(vert, base_verts, base_faces):

    fn = TriNormals(v=base_verts, f=base_faces).reshape((-1, 3))
    vn = VertNormals(v=base_verts, f=base_faces).reshape((-1, 3))

    tree = AabbTree(Mesh(v=base_verts, f=base_faces))
    nearest_tri, nearest_part, nearest_point = tree.nearest(vert, nearest_part=True)
    nearest_tri = nearest_tri.ravel().astype(np.long)
    nearest_part = nearest_part.ravel().astype(np.long)

    nearest_normals = np.zeros_like(vert)

    #nearest_part tells you whether the closest point in triangle abc is in the interior (0), on an edge (ab:1,bc:2,ca:3), or a vertex (a:4,b:5,c:6)
    cl_tri_idxs = np.nonzero(nearest_part == 0)[0].astype(np.int)
    cl_vrt_idxs = np.nonzero(nearest_part > 3)[0].astype(np.int)
    cl_edg_idxs = np.nonzero((nearest_part <= 3) & (nearest_part > 0))[0].astype(np.int)

    nt = nearest_tri[cl_tri_idxs]
    nearest_normals[cl_tri_idxs] = fn[nt]

    nt = nearest_tri[cl_vrt_idxs]
    npp = nearest_part[cl_vrt_idxs] - 4
    nearest_normals[cl_vrt_idxs] = vn[base_faces[nt, npp]]

    nt = nearest_tri[cl_edg_idxs]
    npp = nearest_part[cl_edg_idxs] - 1
    nearest_normals[cl_edg_idxs] += vn[base_faces[nt, npp]]
    npp = np.mod(nearest_part[cl_edg_idxs], 3)
    nearest_normals[cl_edg_idxs] += vn[base_faces[nt, npp]]

    nearest_normals = nearest_normals / (np.linalg.norm(nearest_normals, axis=-1, keepdims=True) + 1.e-10)

    return nearest_point, nearest_normals

def get_collision_loss(garment_verts, body_verts, body_faces, output_collision_percentage=False):

    device = garment_verts.get_device()

    nearest_points, nearest_normals = get_nearest_points_and_normals(garment_verts.cpu().detach().numpy(), body_verts.cpu().detach().numpy(), body_faces)

    torch_nearest_points = torch.from_numpy(nearest_points).to(device)
    torch_nearest_normals = torch.from_numpy(nearest_normals).to(device)

    distance = torch.nn.functional.relu(torch.sum(- (garment_verts - torch_nearest_points) * torch_nearest_normals, axis=1))

    if output_collision_percentage == False:
        return distance.sum()
    else:
        collision_vertices = distance>0
        return distance.sum(), torch.Tensor.float(collision_vertices).mean()

def get_SDF(garment_verts, body_verts, body_faces):

    device = garment_verts.get_device()

    nearest_points, nearest_normals = get_nearest_points_and_normals(garment_verts.cpu().detach().numpy(), body_verts.cpu().detach().numpy(), body_faces)

    torch_nearest_points = torch.from_numpy(nearest_points).to(device)
    torch_nearest_normals = torch.from_numpy(nearest_normals).to(device)

    distance = torch.sum( (garment_verts - torch_nearest_points) * torch_nearest_normals, axis=1)

    return distance


class Accurate_SDF(nn.Module):
    def __init__(self, body_faces):
        super(Accurate_SDF, self).__init__()

        self.body_faces = body_faces

    def forward(self, batch_garment_verts, batch_body_verts):
        batch_size = batch_garment_verts.shape[0]

        per_model_sdf = []

        for i in range(0, batch_size):
            g_verts = batch_garment_verts[i,:,:]
            b_verts = batch_body_verts[i,:,:]

            per_model_sdf.append(get_SDF(g_verts, b_verts, self.body_faces))

        return torch.stack(per_model_sdf)



class Soft_Collision_Loss(nn.Module):
    def __init__(self, body_faces):
        super(Soft_Collision_Loss, self).__init__()

        self.body_faces = body_faces
        

    def forward(self, batch_garment_verts, batch_body_verts, output_collision_percentage=False):
        batch_size = batch_garment_verts.shape[0]

        per_model_collision_loss = []

        if output_collision_percentage == True:
            per_model_collision_percentage = []

        for i in range(0, batch_size):
            g_verts = batch_garment_verts[i,:,:]
            b_verts = batch_body_verts[i,:,:]

            if output_collision_percentage == False:
                per_model_collision_loss.append(get_collision_loss(g_verts, b_verts, self.body_faces))
            else:
                collision_loss, collision_percentage = get_collision_loss(g_verts, b_verts, self.body_faces, output_collision_percentage=True)
                per_model_collision_loss.append(collision_loss)
                per_model_collision_percentage.append(collision_percentage)

        if output_collision_percentage == False:
            return torch.stack(per_model_collision_loss)
        else:
            return torch.stack(per_model_collision_loss), torch.stack(per_model_collision_percentage)

class Estimated_Soft_Collision_Loss(nn.Module):
    def __init__(self, sdf_network):
        super(Estimated_Soft_Collision_Loss, self).__init__()

        self.sdf_network = sdf_network

    def add_latent(self, thetas, betas, points):
        batch_size, num_of_points, dim = points.shape
        points = points.reshape(batch_size * num_of_points, dim)
        latent_inputs = torch.zeros(0).cuda()

        for ind in range(0, batch_size):
            latent_ind = torch.cat([betas[ind], thetas[ind]], 0)
            latent_repeat = latent_ind.expand(num_of_points, -1)
            latent_inputs = torch.cat([latent_inputs, latent_repeat], 0)
        points = torch.cat([latent_inputs, points], 1)
        return points

    def forward(self, thetas, betas, verts):
        batch_size, num_of_points, _ = verts.shape

        latent_verts = self.add_latent(thetas, betas, verts)
        
        sdf_value = self.sdf_network(latent_verts).reshape(batch_size, num_of_points)

        collision_loss = torch.nn.functional.relu(-sdf_value).sum(dim=1)

        return collision_loss