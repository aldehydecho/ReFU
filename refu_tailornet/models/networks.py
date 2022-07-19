import torch.nn as nn
import torch


class FullyConnected(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=1024, num_layers=None):
        super(FullyConnected, self).__init__()
        net = [
            nn.Linear(input_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
        ]
        for i in range(num_layers - 2):
            net.extend([
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(inplace=True),
            ])
        net.extend([
            nn.Linear(hidden_size, output_size),
        ])
        self.net = nn.Sequential(*net)

    def forward(self, x):
        return self.net(x)


class FullyConnected_Reshape(nn.Module):
    def __init__(self, input_size, output_size, output_shape, hidden_size=1024, num_layers=None, drop_prob = 0):
        super(FullyConnected_Reshape, self).__init__()
        net = [
            nn.Linear(input_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(p=drop_prob),
        ]
        for i in range(num_layers - 2):
            net.extend([
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(inplace=True),
            ])
        net.extend([
            nn.Linear(hidden_size, output_size),
        ])
        self.net = nn.Sequential(*net)

        self.output_shape = output_shape

    def forward(self, x):
        x = self.net(x)

        batch_shape = x.shape[0]

        x = torch.reshape(x, [batch_shape, self.output_shape[0], self.output_shape[1]])

        return x

class FullyConnected_SDF_Hybrid_Weight(nn.Module):
    def __init__(self, input_size, middle_output_size, middle_output_shape, hidden_size=1024, num_layers=None, drop_prob = 0):
        super(FullyConnected_SDF_Hybrid_Weight, self).__init__()
        net = [
            nn.Linear(input_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(p=drop_prob),
        ]
        for i in range(num_layers - 2):
            net.extend([
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(inplace=True),
            ])
        net.extend([
            nn.Linear(hidden_size, middle_output_size),
        ])
        self.net = nn.Sequential(*net)

        self.middle_output_shape = middle_output_shape

        weight_net = [nn.Linear(self.middle_output_shape[1]+1, 10), 
                      nn.ReLU(inplace=True),
                      nn.Linear(10, 1)
                      ]
        
        self.weight_net = nn.Sequential(*weight_net)

    def forward(self, x, sdf_value):
        x = self.net(x)

        batch_shape = x.shape[0]

        x = torch.reshape(x, [batch_shape, self.middle_output_shape[0], self.middle_output_shape[1]])

        x = torch.cat((x, sdf_value), -1)

        x = torch.abs(self.weight_net(x))


        return x
