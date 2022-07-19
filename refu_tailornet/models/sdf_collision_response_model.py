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

class SDF_Collsion_Response_Hybrid(nn.Module):
    def __init__(self, sdf_network, hybrid_weight_model, garment_class):
        super(SDF_Collsion_Response_Hybrid, self).__init__()

        self.sdf_network = sdf_network

        self.hybrid_weight_model = hybrid_weight_model

        self.garment_class = garment_class

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

    def forward(self, thetas, betas, gammas, verts, eval = False):

        mlp_thetas, mlp_betas, mlp_gammas = ops.mask_inputs(thetas, betas, gammas, self.garment_class)

        batch_size, num_of_points, _ = verts.shape

        verts.requires_grad_()
        

        latent_verts = self.add_latent(thetas, betas, verts)

        
        sdf_value = self.sdf_network(latent_verts).reshape(batch_size, num_of_points, 1)


        d_points = torch.ones_like(sdf_value, requires_grad=False, device=sdf_value.device)
        
        sdf_gradient = grad(outputs = sdf_value, inputs = verts, grad_outputs=d_points, create_graph=False, retain_graph=True, only_inputs=True)[0]

        normalized_sdf_gradient = F.normalize(sdf_gradient, dim=2)


        hybrid_weight = self.hybrid_weight_model(torch.cat((mlp_thetas, mlp_betas, mlp_gammas), dim=1), sdf_value)

        hybrid_verts = torch.where(sdf_value<0, verts-sdf_value * hybrid_weight * normalized_sdf_gradient, verts)
        


        hybrid_latent_verts = self.add_latent(thetas, betas, hybrid_verts)

        hybrid_sdf_value = self.sdf_network(hybrid_latent_verts).reshape(batch_size, num_of_points, 1)

        return hybrid_verts, hybrid_sdf_value, sdf_value, hybrid_weight


