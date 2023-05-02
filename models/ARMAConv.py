import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import ARMAConv
from torch_geometric.nn import global_mean_pool


class ARMANet(torch.nn.Module):
    def __init__(self, in_dim,
                 hid_dim,
                 out_dim,
                 num_stacks=1,
                 num_layers=1,
                 activation=None,
                 dropout=0.0):
        super(ARMANet, self).__init__()

        self.conv1 = ARMAConv(in_channels=in_dim,
                              out_channels=hid_dim,
                              num_stacks=num_stacks,
                              num_layers=num_layers,
                              activation=activation,
                              dropout=dropout)

        self.conv2 = ARMAConv(in_channels=hid_dim,
                              out_channels=hid_dim,
                              num_stacks=num_stacks,
                              num_layers=num_layers,
                              activation=activation,
                              dropout=dropout)

        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(hid_dim, out_dim)


    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, data.batch)
        x = self.dropout(x)
        x = self.fc(x)
        return x
