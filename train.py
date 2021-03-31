import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, help='which dataset to use')
parser.add_argument('--config', type=str, help='configuration file')
parser.add_argument('--force_step', type=int, default=0, help='force the step length to be some number')
parser.add_argument('--gpu', type=int, default=0, help='which gpu to use')
parser.add_argument('--store_result', type=str, default='', help='store the result for tuning')
args = parser.parse_args()

import os
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

import torch
import dgl
import colorama
import numpy as np
from events import Events
from tqdm import tqdm
import time
import datetime
from utils import *
from models import *

data = Events(args.data)

max_step = data.ts_test + data.ts_val if args.force_step == 0 else args.force_step
if not os.path.isdir('models'):
    os.mkdir('models')
save_path = 'models/' + args.data + str(datetime.datetime.now())[:19] + '.pkl'

emb_conf, gen_conf, train_conf = parse_train_config(args.config)
max_history = max([int(h) for h in gen_conf['history'].split(' ')])

# generate graph, here we build the full graph because a node at time t will only aggregate from neighbors in the past
g = None
print('Constructing graph...')
total_length = data.ts_train
if args.force_step > 0:
    total_length += data.ts_val + data.ts_test
with tqdm(total=total_length) as pbar:
    for ts in range(0, data.ts_train):
        event_dict = data.get_hetero_dict(ts, emb_conf['history'])
        if g is None:
            add_virtual_relation(event_dict, data.num_entity, data.num_relation)
            g = dgl.heterograph(event_dict)
            g.remove_nodes(data.num_entity, 'entity')
        else:
            add_edges_from_dict(g, event_dict)
        pbar.update(1)
    if args.force_step > 0:
        # since a specific step is selected, it is possible to use ground truth in val or test set
        for ts in range(data.ts_train, data.ts_train + data.ts_val + data.ts_test):
            event_dict = data.get_hetero_dict(ts, emb_conf['history'])
            add_edges_from_dict(g, event_dict)
        pbar.update(1)

model = FixStepModel(emb_conf, gen_conf, train_conf, g, data.num_entity, data.num_relation, max_step).cuda()
max_mrr = 0
max_e = 0
for e in range(train_conf['epoch']):
    torch.cuda.empty_cache()
    print('epoch {:d}:'.format(e))
    with tqdm(total=data.ts_train + data.ts_val - train_conf['fwd'] - max_history - max_step) as pbar:
        # training
        train_rank_unf = list()
        for ts in range(train_conf['fwd'] + max_history + max_step, data.ts_train):
            if gen_conf['copy'] > 0:
                batches = data.get_batches(ts, train_conf['batch_size'], require_mask=False, copy_mask_ts=max_step)
                for b in range(len(batches[0])):
                    _, rank_unf, _ = model.step(batches[0][b], batches[1][b], batches[2][b], ts, filter_mask=None, copy_mask=batches[4][b], train=True)
                    with torch.no_grad():
                        train_rank_unf.append(rank_unf)
            else:
                batches = data.get_batches(ts, train_conf['batch_size'], require_mask=False)
                for b in range(len(batches[0])):
                    _, rank_unf, _ = model.step(batches[0][b], batches[1][b], batches[2][b], ts, filter_mask=None, train=True)
                    with torch.no_grad():
                        train_rank_unf.append(rank_unf)
            pbar.update(1)
        # validation
        total_rank_unf = list()
        total_rank_fil = list()
        with torch.no_grad():
            for ts in range(data.ts_train, data.ts_train + data.ts_val):
                if gen_conf['copy'] > 0:
                    batches = data.get_batches(ts, -1, require_mask=True, copy_mask_ts=max_step)
                    loss, rank_unf, rank_fil = model.step(batches[0][0], batches[1][0], batches[2][0], ts, filter_mask=batches[3][0], copy_mask=batches[4][0], train=False)
                else:
                    batches = data.get_batches(ts, -1, require_mask=True)
                    loss, rank_unf, rank_fil = model.step(batches[0][0], batches[1][0], batches[2][0], ts, filter_mask=batches[3][0], train=False)
                total_rank_unf.append(rank_unf)
                total_rank_fil.append(rank_fil)
                pbar.update(1)
            total_rank_unf = torch.cat(total_rank_unf)
            total_rank_fil = torch.cat(total_rank_fil)
    with torch.no_grad():
        train_rank_unf = torch.cat(train_rank_unf)
    print('\ttrain raw MRR:      {:.4f}'.format(mrr(train_rank_unf)))
    print('\tvalid raw MRR:      {:.4f} hit3: {:.4f} hit10: {:.4f}'.format(mrr(total_rank_unf), hit3(total_rank_unf), hit10(total_rank_unf)))
    print('\tvalid filtered MRR: {:.4f} hit3: {:.4f} hit10: {:.4f}'.format(mrr(total_rank_fil), hit3(total_rank_fil), hit10(total_rank_fil)))
    if mrr(total_rank_fil) > max_mrr:
        max_mrr = mrr(total_rank_fil)
        max_e = e
        torch.save(model.state_dict(), save_path)
# testing
print(colorama.Fore.RED + 'Testing...'+ colorama.Style.RESET_ALL)
if max_e > 0:
    print(colorama.Fore.RED + 'Loading epoch {:d} with filtered MRR {:.4f}'.format(max_e, max_mrr) + colorama.Style.RESET_ALL)
    model.load_state_dict(torch.load(save_path))
rank_fil_l = list()
with tqdm(total=data.ts_test) as pbar:
    with torch.no_grad():
        total_rank_unf = list()
        total_rank_fil = list()
        for ts in range(data.ts_train + data.ts_val, data.ts_train + data.ts_val + data.ts_test):
            rank_unf_e = list()
            rank_fil_e = list()
            if gen_conf['copy'] > 0:
                batches = data.get_batches(ts, train_conf['batch_size'], require_mask=True, copy_mask_ts=max_step)
                for b in range(len(batches[0])):
                    loss, rank_unf, rank_fil = model.step(batches[0][0], batches[1][0], batches[2][0], ts, filter_mask=batches[3][0], copy_mask=batches[4][0], train=False)
                    rank_unf_e.append(rank_unf)
                    rank_fil_e.append(rank_fil)
            else:
                batches = data.get_batches(ts, train_conf['batch_size'], require_mask=True)
                for b in range(len(batches[0])):
                    loss, rank_unf, rank_fil = model.step(batches[0][0], batches[1][0], batches[2][0], ts, filter_mask=batches[3][0], train=False)
                    rank_unf_e.append(rank_unf)
                    rank_fil_e.append(rank_fil)
            rank_unf_e = torch.cat(rank_unf_e)
            rank_fil_e = torch.cat(rank_fil_e)
            total_rank_unf.append(rank_unf_e.cpu())
            total_rank_fil.append(rank_fil_e.cpu())
            rank_fil_l.append(mrr(total_rank_fil[-1]))
            pbar.update(1)
        total_rank_unf = torch.cat(total_rank_unf)
        total_rank_fil = torch.cat(total_rank_fil)
print(colorama.Fore.RED + '\traw MRR:      {:.4f} hit3: {:.4f} hit10: {:.4f}'.format(mrr(total_rank_unf), hit3(total_rank_unf), hit10(total_rank_unf)) + colorama.Style.RESET_ALL)
print(colorama.Fore.RED + '\tfiltered MRR: {:.4f} hit3: {:.4f} hit10: {:.4f}'.format(mrr(total_rank_fil), hit3(total_rank_fil), hit10(total_rank_fil)) + colorama.Style.RESET_ALL)
if args.force_step > 0:
    print(colorama.Fore.RED + '\tfiltered MRR at each timestamp: '+ '\t'.join(str(float(fil)) for fil in rank_fil_l) + colorama.Style.RESET_ALL)
# if args.store_result != "":
#     with open(args.store_result, encoding="utf-8", mode="a") as f:
#         f.write('{:.2f}\t{:.2f}\t{:.5f}\t{:.4f}\n'.format(args.copy,args.dropout, args.lr, mrr(total_rank_fil)))