import torch.nn
from torch_geometric.nn.conv import GCNConv
from config import *
import torch.nn.functional as F


class GCN(torch.nn.Module):
    def __init__(self, num_node_features, num_classes, hidden_dim):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_node_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.FC = torch.nn.Linear(hidden_dim, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, training=self.training, p=args.dropout)
        x = self.conv2(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, training=self.training, p=args.dropout)
        x = self.conv3(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, training=self.training, p=args.dropout)
        x = self.FC(x)
        return F.log_softmax(x, dim=-1)
