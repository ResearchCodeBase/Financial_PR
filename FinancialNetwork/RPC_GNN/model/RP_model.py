import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv

class GraphGCN(torch.nn.Module):
    def __init__(self, in_channels,data):
        super(GraphGCN, self).__init__()
        self.conv1 = GCNConv(data.num_features, 32)
        self.conv2 = GCNConv(32, 2)


    def forward(self, x, edge_index,edge_weight):
        x = self.conv1(x, edge_index,edge_weight)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index,edge_weight)
        return x





