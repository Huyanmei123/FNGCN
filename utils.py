import pickle as pkl
import sys
from torch_geometric import utils
import torch
import numpy as np
from config import *


def load_data(dataset_str):
    if args.type == 'Binary':
        names = [args.feature_type, 'graph', 'train_mask', 'test_mask', 'labeled']
    else:
        names = [args.feature_type, 'graph', 'SIR']
    objects = []
    for i in range(len(names)):
        with open("../data/{}/{}-{}.{}".format(dataset_str, dataset_str, names[i], names[i]),
                  'rb') as f:
            if sys.version_info > (3, 0):

                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))
    if args.type == 'Binary':
        feature, graph, train_mask, test_mask, label = tuple(objects)
    else:
        feature, graph, label = tuple(objects)
    # Converts a networkx.Graph or networkx.DiGraph to a torch_geometric.data.Data instance.
    data = utils.from_networkx(graph)

    data.x = torch.tensor(np.array(feature), dtype=torch.float32)
    label = torch.tensor(np.array(label), dtype=torch.long)

    data.train_mask = train_mask
    data.test_mask = test_mask
    data.label = label
    print(data)
    return data
