import torch
from torch_geometric.nn import SAGEConv, GATConv, GCNConv

from layers.gcn_conv import GCNConvCache
from layers.sage_conv import SAGEConvCache

class ConvLayer(torch.nn.Module):
    def __init__(self, gnn, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

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
        self.relu = torch.nn.ReLU()
        
    def forward(self, x, edge_index, edge_type=None):
        # edge_type is not needed in SAGE GAT or GCN
        # but we still take the input for compatibility with heterogeneous GNNs, and simply ignore it here
        return self.relu(self.gnn(x, edge_index))


