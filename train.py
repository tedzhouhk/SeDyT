import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, help='which dataset to use')
parser.add_argument('--fwd', type=int, default=10, help='how many time stamps forwarded in training')
parser.add_argument('-e', '--epoch', type=int, default=10, help='number of epochs to train')
parser.add_argument('--dim', type=int, default=512, help='dimension of hidden features')
parser.add_argument('--dropout', type=float, help='dropout rate')
parser.add_argument('--lr', type=float, default=0.001,help='learning rate')
parser.add_argument('--gpu', type=int, default=1, help='which gpu to use')
parser.add_argument('--verbose', default=0, action='store_true', help='whether to prinmt verbose output')
parser.add_argument('--network_type', default='single', type=str, help='single, multi, or hybird')
parser.add_argument('--history', type=int, nargs='+', default=[31, 15, 7, 3, 2, 1, 0], help='previous steps to look at')
parser.add_argument('--attention_head', type=int, default=8, help='number of attention heads')
parser.add_argument('--layer', type=int, default=1, help='number of layers')
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

max_step = data.ts_test

ent_emb = torch.load('data/' + args.data + '/ent_emb.pt', map_location=torch.device('cuda:0'))
obj_rel_emb = torch.load('data/' + args.data + '/obj_rel_emb.pt', map_location=torch.device('cuda:0'))
sub_rel_emb = torch.load('data/' + args.data + '/sub_rel_emb.pt', map_location=torch.device('cuda:0'))

dim_in = ent_emb.shape[2] * len(args.history)

if args.network_type == 'single':
    model = FixStepAttentionModel(dim_in, args.dim, obj_rel_emb.shape[1], ent_emb.shape[1], sub_rel_emb, obj_rel_emb, dropout=args.dropout, h_att=args.attention_head, num_l=args.layer, lr=args.lr).cuda()

    for e in range(args.epoch):
        print('epoch {:d}:'.format(e))
        with tqdm(total=data.ts_train + data.ts_val - args.fwd - max(args.history) - max_step) as pbar:
            # training
            for ts in range(args.fwd + max(args.history) + max_step, data.ts_train):
                hid = extract_emb(ent_emb, args.history, ts - max_step)
                batches = data.get_batches(ts, -1, require_mask=False)
                model.step(hid, batches[0][0], batches[1][0], batches[2][0], filter_mask=None, train=True)
                pbar.update(1)
            # validation
            total_rank_unf = list()
            total_rank_fil = list()
            with torch.no_grad():
                for ts in range(data.ts_train, data.ts_train + data.ts_val):
                    hid = extract_emb(ent_emb, args.history, ts - max_step)
                    batches = data.get_batches(ts, -1, require_mask=True)
                    loss, rank_unf, rank_fil = model.step(hid, batches[0][0], batches[1][0], batches[2][0], batches[3][0], train=False)
                    total_rank_unf.append(rank_unf)
                    total_rank_fil.append(rank_fil)
                    pbar.update(1)
                total_rank_unf = torch.cat(total_rank_unf)
                total_rank_fil = torch.cat(total_rank_fil)
        print('\traw MRR:      {:.4f} hit3: {:.4f} hit10: {:.4f}'.format(mrr(total_rank_unf), hit3(total_rank_unf), hit10(total_rank_unf)))
        print('\tfiltered MRR: {:.4f} hit3: {:.4f} hit10: {:.4f}'.format(mrr(total_rank_fil), hit3(total_rank_fil), hit10(total_rank_fil)))
        # # testing
        # with tqdm(total=data.ts_test) as pbar:
        #     with torch.no_grad():
        #         total_rank_unf = list()
        #         total_rank_fil = list()
        #         for ts in range(data.ts_train + data.ts_val, data.ts_train + data.ts_val + data.ts_test):
        #             hid = extract_emb(ent_emb, args.history, ts - max_step)
        #             batches = data.get_batches(ts, -1, require_mask=True)
        #             loss, rank_unf, rank_fil = model.step(hid, batches[0][0], batches[1][0], batches[2][0], batches[3][0], train=False)
        #             total_rank_unf.append(rank_unf)
        #             total_rank_fil.append(rank_fil)
        #             pbar.update(1)
        #         total_rank_unf = torch.cat(total_rank_unf)
        #         total_rank_fil = torch.cat(total_rank_fil)
        # print('Test result:')
        # print('\traw MRR:      {:.4f} hit3: {:.4f} hit10: {:.4f}'.format(mrr(total_rank_unf), hit3(total_rank_unf), hit10(total_rank_unf)))
        # print('\tfiltered MRR: {:.4f} hit3: {:.4f} hit10: {:.4f}'.format(mrr(total_rank_fil), hit3(total_rank_fil), hit10(total_rank_fil)))
    # testing
    with tqdm(total=data.ts_test) as pbar:
        with torch.no_grad():
            total_rank_unf = list()
            total_rank_fil = list()
            for ts in range(data.ts_train + data.ts_val, data.ts_train + data.ts_val + data.ts_test):
                hid = extract_emb(ent_emb, args.history, ts - max_step)
                batches = data.get_batches(ts, -1, require_mask=True)
                loss, rank_unf, rank_fil = model.step(hid, batches[0][0], batches[1][0], batches[2][0], batches[3][0], train=False)
                total_rank_unf.append(rank_unf)
                total_rank_fil.append(rank_fil)
                pbar.update(1)
            total_rank_unf = torch.cat(total_rank_unf)
            total_rank_fil = torch.cat(total_rank_fil)
    print('Test result:')
    print('\traw MRR:      {:.4f} hit3: {:.4f} hit10: {:.4f}'.format(mrr(total_rank_unf), hit3(total_rank_unf), hit10(total_rank_unf)))
    print('\tfiltered MRR: {:.4f} hit3: {:.4f} hit10: {:.4f}'.format(mrr(total_rank_fil), hit3(total_rank_fil), hit10(total_rank_fil)))