import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, help='which dataset to use')
parser.add_argument('--fwd', type=int, default=10, help='how many time stamps forwarded in training')
parser.add_argument('-e', '--epoch', type=int, default=10, help='number of epochs to train')
parser.add_argument('--dim', type=int, default=128, help='dimension of hidden features')
parser.add_argument('--dim_e', type=int, default=128, help='dimension of entity attributes')
parser.add_argument('--dim_r', type=int, default=128, help='dimension of relation attributes')
parser.add_argument('--dropout', type=float, help='dropout rate')
parser.add_argument('--lr', type=float, default=0.01,help='learning rate')
parser.add_argument('--batch_size', type=int, default=1024, help='batch size used in training')
parser.add_argument('--gpu', type=int, default=0, help='which gpu to use')
parser.add_argument('--verbose', default=0, action='store_true', help='whether to prinmt verbose output')
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

for e in range(args.epoch):
    print('epoch {:d}:'.format(e))
    g = None
    model.train()
    t_prep = 0
    t_forw = 0
    t_back = 0
    t_grap = 0
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
                t_s = time.time()
                batches = data.get_batches(ts, args.batch_size, require_mask=False)
                t_prep += time.time() - t_s
                for sub, obj, rel in zip(batches[0], batches[1], batches[2]):
                    optimizer.zero_grad()
                    t_s = time.time()
                    loss, rank_unf, rank_fil = model.forward_and_loss(sub, obj, rel, g)
                    t_forw += time.time() - t_s
                    t_s = time.time()
                    loss.backward()
                    optimizer.step()
                    t_back += time.time() - t_s
            t_s = time.time()
            add_edges_from_dict(g, event_dict)
            t_grap += time.time() - t_s
            pbar.update(1)
        total_rank_unf = list()
        total_rank_fil = list()
        for ts in range(data.ts_train, data.ts_train + data.ts_val):
            event_dict = data.get_hetero_dict(ts)
            batches = data.get_batches(ts, args.batch_size)
            for sub, obj, rel, mask in zip(batches[0], batches[1], batches[2], batches[3]):
                optimizer.zero_grad()
                t_s = time.time()
                loss, rank_unf, rank_fil = model.forward_and_loss(sub, obj, rel, g, mask)
                t_forw += time.time() - t_s
                t_s = time.time()
                loss.backward()
                torch.nn.utils.clip_grad_norm(model.parameters(), 5)
                optimizer.step()
                t_back += time.time() - t_s
                total_rank_unf.append(rank_unf.detach().clone())
                total_rank_fil.append(rank_fil.detach().clone())
                # print(mrr(total_rank_fil[-1]))
            add_edges_from_dict(g, event_dict)
            pbar.update(1)
    with torch.no_grad():
        total_rank_unf = torch.cat(total_rank_unf)
        total_rank_fil = torch.cat(total_rank_fil)
        if args.verbose:
            print('\tprep: {:.2f}s forw: {:.2f}s back: {:.2f}s grap: {:.2f}s'.format(t_prep, t_forw, t_back, t_grap))
        print('\traw MRR: {:.4f} hit3: {:.4f} hit10: {:.4f}'.format(mrr(total_rank_unf), hit3(total_rank_unf), hit10(total_rank_unf)))
        print('\tfiltered MRR: {:.4f} hit3: {:.4f} hit10: {:.4f}'.format(mrr(total_rank_fil), hit3(total_rank_fil), hit10(total_rank_fil)))
    
print('Optimization finished!')
print('Generating entity and relation embeddings...')
with torch.no_grad():
    sub_rel_emb = model.mods['subject_relation_emb'].weight.clone().detach().cpu()
    obj_rel_emb = model.mods['object_relation_emb'].weight.clone().detach().cpu()
    entity_emb = list()
    g = None
    model.eval()
    with tqdm(total=data.ts_train + data.ts_val + data.ts_test) as pbar:
        for ts in range(0, data.ts_train):
            event_dict = data.get_hetero_dict(ts)
            if g is None:
                add_virtual_relation(event_dict, data.num_entity, data.num_relation)
                g = dgl.heterograph(event_dict).to('cuda:0')
                g.remove_nodes(data.num_entity, 'entity')
            else:
                add_edges_from_dict(g, event_dict)
            entity_emb.append(model.get_entity_embedding(g).clone().detach().cpu())
            pbar.update(1)
        for ts in range(data.ts_train, data.ts_train + data.ts_val):
            event_dict = data.get_hetero_dict(ts)
            add_edges_from_dict(g, event_dict)
            entity_emb.append(model.get_entity_embedding(g).clone().detach().cpu())
            pbar.update(1)
        for ts in range(data.ts_train + data.ts_val, data.ts_train + data.ts_val + data.ts_test):
            event_dict = data.get_hetero_dict(ts)
            add_edges_from_dict(g, event_dict)
            entity_emb.append(model.get_entity_embedding(g).clone().detach().cpu())
            pbar.update(1)
    entity_emb = torch.stack(entity_emb)
    if not os.path.isdir('data'):
        os.mkdir('data')
    if not os.path.isdir('data/' + args.data):
        os.mkdir('data/' + args.data)
    torch.save(sub_rel_emb, 'data/' + args.data + '/sub_rel_emb.pt')
    torch.save(obj_rel_emb, 'data/' + args.data + '/obj_rel_emb.pt')
    torch.save(entity_emb, 'data/' + args.data + '/ent_emb.pt')
            
