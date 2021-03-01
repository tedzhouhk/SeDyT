import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, help='which dataset to use')
parser.add_argument('--fwd', type=int, default=10, help='how many time stamps forwarded in training')
parser.add_argument('-e', type=int, default=10, help='number of epochs to train')
parser.add_argument('--dim', type=int, default=128, help='dimension of hidden features')
parser.add_argument('--dim_e', type=int, default=128, help='dimension of entity attributes')
parser.add_argument('--dim_r', type=int, default=128, help='dimension of relation attributes')
parser.add_argument('--dropout', type=float, help='dropout rate')
parser.add_argument('--lr', type=float, default=0.01,help='learning rate')
parser.add_argument('--batch_size', type=int, default=1024, help='batch size used in training')
parser.add_argument('--gpu', type=int, default=1, help='which gpu to use')
args = parser.parse_args()

import os
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

import torch
import dgl
import numpy as np
from events import Events
from tqdm import tqdm
import time
from utils import *
from models import *

data = Events(args.data)

model = PreTrainModel(args.dim_e, args.dim, args.dim_r, data.num_relation, data.num_entity, args.dropout).cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

for e in range(args.e):
    print('epoch {:d}:'.format(e))
    g = None
    model.train()
    with tqdm(total=data.ts_train + data.ts_val) as pbar:
        for ts in range(0, data.ts_train):
            event_dict = data.get_hetero_dict(ts)
            if ts < args.fwd:
                if g is None:
                    add_virtual_relation(event_dict, data.num_entity, data.num_relation)
                    g = dgl.heterograph(event_dict).to('cuda:0')
                    g.remove_nodes(data.num_entity, 'entity')
                else:
                    add_edges_from_dict(g, event_dict)
                pbar.update(1)
                continue
            else:
                import pdb; pdb.set_trace()
                batches = data.get_batches(ts, args.batch_size, require_mask=False)
                for sub, obj, rel in zip(batches[0], batches[1], batches[2]):
                    optimizer.zero_grad()
                    loss, rank_unf, rank_fil = model.forward_and_loss(sub, obj, rel, g)
                    loss.backward()
                    optimizer.step()
            add_edges_from_dict(g, event_dict)
            pbar.update(1)
        total_rank_unf = list()
        total_rank_fil = list()
        model.eval()
        for ts in range(data.ts_train, data.ts_train + data.ts_val):
            event_dict = data.get_hetero_dict(ts)
            batches = data.get_batches(ts, args.batch_size)
            for sub, obj, rel, mask in zip(batches[0], batches[1], batches[2], batches[3]):
                loss, rank_unf, rank_fil = model.forward_and_loss(sub, obj, rel, g, mask)
                total_rank_unf.append(rank_unf)
                total_rank_fil.append(rank_fil)
            add_edges_from_dict(g, event_dict)
            pbar.update(1)
    with torch.no_grad():
        total_rank_unf = torch.cat(total_rank_unf)
        total_rank_fil = torch.cat(total_rank_fil)
        print('\traw MRR: {:.4f} hit3: {:.4f} hit10: {:.4f}'.format(mrr(total_rank_unf), hit3(total_rank_unf), hit10(total_rank_unf)))
        print('\tfiltered MRR: {:.4f} hit3: {:.4f} hit10: {:.4f}'.format(mrr(total_rank_fil), hit3(total_rank_fil), hit10(total_rank_fil)))