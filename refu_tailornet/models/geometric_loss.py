import torch.nn as nn
import torch

class Laplace_Mean_Euclidean_Loss(nn.Module):
    def __init__(self, neighbour, degrees, max_degree, point_num):
        super(Laplace_Mean_Euclidean_Loss, self).__init__()

        if type(neighbour) == torch.nn.parameter.Parameter:
            self.neighbour = neighbour
        else:
            self.neighbour = torch.nn.Parameter(torch.from_numpy(neighbour).long(), requires_grad=False)
        if type(degrees) == torch.nn.parameter.Parameter:
            self.degrees = degrees
        else:
            self.degrees = torch.nn.Parameter(torch.from_numpy(degrees).int(), requires_grad=False)

        self.max_degree = max_degree

        self.point_num = point_num

    def forward(self, predict_points, gt_points):

        batch = predict_points.shape[0]

        zeros = torch.zeros(batch, 1, 3).to(predict_points.device)

        padded_predict_points = torch.cat([predict_points, zeros], dim=1)
        padded_gt_points = torch.cat([gt_points, zeros], dim=1)

        gt_laplace = gt_points*(self.degrees.view(1, self.point_num, 1).repeat(batch, 1, 3)) - padded_gt_points[:, self.neighbour, :].sum(dim=2)

        predict_laplace = predict_points*(self.degrees.view(1, self.point_num, 1).repeat(batch, 1, 3)) - padded_predict_points[:, self.neighbour, :].sum(dim=2)

        loss = torch.sqrt(torch.pow(predict_laplace - gt_laplace, 2).sum(2)).sum(1).mean()

        return loss 

class Geometric_Mean_Euclidean_Loss(nn.Module):
    def __init__(self):
        super(Geometric_Mean_Euclidean_Loss, self).__init__()

    def forward(self, predict_points, gt_points):
        loss = (predict_points - gt_points).pow(2).sum(2).sqrt().mean()

        return loss

class Per_Layer_Mean_Euclidean_Loss(nn.Module):
    def __init__(self, decoder_layers, vertex_num):
        super(Per_Layer_Mean_Euclidean_Loss, self).__init__()

        self.vertex_num = vertex_num

        temp_mapping_list = [torch.nn.Parameter(torch.from_numpy(layer[-1]).long(), requires_grad=False) for layer in decoder_layers]
        temp_mapping_list.pop(0)

        temp_mapping_list.append(temp_mapping_list[-1])

        self.center_mapping_list = torch.nn.ParameterList(temp_mapping_list)

    def forward(self, per_layer_predict, gt_points):

        loss = []

        for i, predict in enumerate(per_layer_predict):
            loss.append(torch.sqrt(torch.pow(predict-gt_points[:,self.center_mapping_list[i],:], 2).sum(2)).mean()*self.vertex_num)

        loss = torch.stack(loss).mean()

        return loss





