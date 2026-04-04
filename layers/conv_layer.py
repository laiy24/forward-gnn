import torch
from torch_geometric.nn import SAGEConv, GATConv, GCNConv

from layers.gcn_conv import GCNConvCache
from layers.sage_conv import SAGEConvCache

class ConvLayer(torch.nn.Module):
    def __init__(self, gnn, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.relu = torch.nn.ReLU()

        if "Cached".lower() in gnn.lower():
            if "SAGE" in gnn:
                self.gnn = SAGEConvCache(in_channels=in_channels, out_channels=out_channels, aggr="mean")
            elif "GCN" in gnn:
                self.gnn = GCNConvCache(in_channels=in_channels, out_channels=out_channels)
            else:
                raise ValueError(f"wrong gnn config: {gnn}")
        elif "SAGE" in gnn:
            self.gnn = SAGEConv(in_channels=in_channels, out_channels=out_channels, aggr="mean")
        elif "GCN" in gnn:
            self.gnn = GCNConv(in_channels=in_channels, out_channels=out_channels)
        elif "GAT" in gnn:
            heads = 4
            assert out_channels % heads == 0, (out_channels, heads)
            self.gnn = GATConv(in_channels=in_channels, out_channels=out_channels // heads, heads=heads)
        else:
            raise ValueError(f"wrong gnn config: {gnn}")
    def forward(self, x, edge_index):
        return self.relu(self.gnn(x, edge_index))


