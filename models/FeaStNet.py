import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import FeaStConv
from torch_geometric.nn import global_mean_pool

class FeaStNet(torch.nn.Module):
    def __init__(self, in_channels, num_classes, heads, t_inv=True):
        super(FeaStNet, self).__init__()

        self.fc0 = nn.Linear(in_channels, 16)
        self.conv1 = FeaStConv(16, 32, heads=heads, t_inv=t_inv)
        self.conv2 = FeaStConv(32, 64, heads=heads, t_inv=t_inv)
        self.fc1 = nn.Linear(64, num_classes)

        self.reset_parameters()

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.fc0(x))
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, data.batch)
        x = F.dropout(x, training=self.training)
        x = self.fc1(x)
        return x