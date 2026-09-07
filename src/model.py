# -*- coding: utf-8 -*-
"""
The GCN architecture, identical across all five original scripts and
now defined exactly once.
"""

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GCN(torch.nn.Module):
    def __init__(self, num_nodes, embedding_dim, hidden_channels=32, dropout=0.2):
        super().__init__()
        self.dropout = dropout
        self.embedding = torch.nn.Embedding(num_nodes, embedding_dim)
        self.conv1 = GCNConv(embedding_dim, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.out_head = torch.nn.Linear(hidden_channels, 2)

    def forward(self, data):
        x = self.embedding(data.x)
        x = F.relu(self.conv1(x, data.edge_index, data.edge_attr))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.conv2(x, data.edge_index, data.edge_attr))
        return self.out_head(x)
