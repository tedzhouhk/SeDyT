import argparse
import torch
import dgl
import numpy as np
from events import Events
from utils import *


parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, help='which dataset to use')
parser.add_argument('--fwd', type=int, default=10, help='how many time stamps forwarded in training')
parser.add_argument('-e', type=int, default=10, help='number of epochs to train')
args = parser.parse_args()

data = Events(args.data)

for e in range(args.e):
    g = None
    for ts in range(0, data.ts_train):
        event_dict = data.get_hetero_dict(ts)
        if g is None:
            add_virtual_relation(event_dict, data.num_entity, data.num_relation)
            g = dgl.heterograph(event_dict)
            g.remove_nodes(data.num_entity, 'entity')
        else:
            add_edges_from_dict(g, event_dict)
        if ts < args.fwd:
            continue
        else:
            import pdb; pdb.set_trace()