import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool

from torch_geometric.nn import ARMAConv
from torch_geometric.nn.conv import GATConv
from torch_geometric.nn.conv import FeaStConv
from torch_geometric.nn.conv import ResGatedGraphConv
from torch_geometric.nn.conv import TransformerConv
from torch_geometric.nn import GCNConv


class ARMANet(torch.nn.Module):
    def __init__(self, in_channels,
                 hidden_channels,
                 num_classes,
                 num_stacks=1,
                 num_layers=1):
        super(ARMANet, self).__init__()

        self.conv1 = ARMAConv(in_channels=in_channels,
                              out_channels=hidden_channels,
                              num_stacks=num_stacks,
                              num_layers=num_layers)

        self.conv2 = ARMAConv(in_channels=hidden_channels,
                              out_channels=64,
                              num_stacks=num_stacks,
                              num_layers=num_layers)

        self.fc1 = nn.Linear(64, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, data.batch)
        x = F.dropout(x, training=self.training)
        x = self.fc1(x)
        return x


class GATNet(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels=512, num_classes=2):
        super(GATNet, self).__init__()

        # self.fc0 = nn.Linear(in_channels, hidden_channels)
        self.conv1 = GATConv(in_channels, hidden_channels)
        self.conv2 = GATConv(hidden_channels, 64)
        self.fc1 = nn.Linear(64, num_classes)

        self.reset_parameters()

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # x = F.relu(self.fc0(x))
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, data.batch)
        x = F.dropout(x, training=self.training)
        x = self.fc1(x)
        return x


class FeaStNet(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels=512, num_classes=2, heads=1, t_inv=True):
        super(FeaStNet, self).__init__()

        # self.fc0 = nn.Linear(in_channels, hidden_channels)
        self.conv1 = FeaStConv(in_channels, hidden_channels, heads=heads, t_inv=t_inv)
        self.conv2 = FeaStConv(hidden_channels, 64, heads=heads, t_inv=t_inv)
        self.fc1 = nn.Linear(64, num_classes)

        self.reset_parameters()

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # x = F.relu(self.fc0(x))
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, data.batch)
        x = F.dropout(x, training=self.training)
        x = self.fc1(x)
        return x


class RGGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels=512, num_classes=2):
        super(RGGCN, self).__init__()

        # self.fc0 = nn.Linear(in_channels, hidden_channels)
        self.conv1 = ResGatedGraphConv(in_channels, hidden_channels)
        self.conv2 = ResGatedGraphConv(hidden_channels, 64)
        self.fc1 = nn.Linear(64, num_classes)

        self.reset_parameters()

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # x = F.relu(self.fc0(x))
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, data.batch)
        x = F.dropout(x, training=self.training)
        x = self.fc1(x)
        return x


class UniMP(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels=512, num_classes=2):
        super(UniMP, self).__init__()
        # self.fc0 = nn.Linear(in_channels, hidden_channels)
        self.conv1 = TransformerConv(in_channels, hidden_channels)
        self.conv2 = TransformerConv(hidden_channels, 64)
        self.fc1 = nn.Linear(64, num_classes)

        self.reset_parameters()

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        # x = F.relu(self.fc0(x))
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, data.batch)
        x = F.dropout(x, training=self.training)
        x = self.fc1(x)
        return x


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels=512, num_classes=2):
        super().__init__()

        # self.fc0 = nn.Linear(in_channels, hidden_channels)
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, 64)
        self.fc1 = nn.Linear(64, num_classes)

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        batch = data.batch

        # x = F.relu(self.fc0(x))
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = global_mean_pool(x, batch)
        x = F.dropout(x, training=self.training)
        x = self.fc1(x)
        return x
