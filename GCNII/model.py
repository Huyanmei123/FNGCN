import torch.nn
from torch import nn
from torch_geometric.nn.conv import GCN2Conv
from config import *
import torch.nn.functional as F
from torch.nn import Conv2d
from torch.nn import MaxPool2d
from torch.nn import Linear


class GCNII(torch.nn.Module):
    def __init__(self, num_node_features, num_classes, hidden_dim, n_layers=args.layers):
        super(GCNII, self).__init__()
        self.convs = nn.ModuleList()
        for _ in range(n_layers):
            self.convs.append(GCN2Conv(channels=hidden_dim, alpha=args.alpha, theta=args.theta, layer=_ + 1))
        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(num_node_features, hidden_dim))
        self.fcs.append(nn.Linear(hidden_dim, num_classes))
        self.params1 = list(self.convs.parameters())
        self.params2 = list(self.fcs.parameters())


    def forward(self, data):

        x, edge_index = data.x, data.edge_index
        _layers = []
        layer_inner = self.fcs[0](x)
        _layers.append(layer_inner)
        for i, con in enumerate(self.convs):
            layer_inner = F.elu(con(layer_inner, _layers[0], edge_index))
            layer_inner = F.dropout(layer_inner, args.dropout, training=self.training)
        layer_inner = self.fcs[-1](layer_inner)
        return layer_inner.log_softmax(dim=-1)
