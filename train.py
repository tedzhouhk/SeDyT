import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, help='which dataset to use')
parser.add_argument('--fwd', type=int, default=10, help='how many time stamps forwarded in training')
parser.add_argument('-e', '--epoch', type=int, default=10, help='number of epochs to train')
parser.add_argument('--dim', type=int, default=512, help='dimension of hidden features')
parser.add_argument('--dropout', type=float, help='dropout rate')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--gpu', type=int, default=1, help='which gpu to use')
parser.add_argument('--network_type', default='single', type=str, help='single, multi, or hybird')
parser.add_argument('--history', type=int, nargs='+', default=[31, 15, 7, 3, 2, 1, 0], help='previous steps to look at')
parser.add_argument('--attention_head', type=int, default=8, help='number of attention heads')
parser.add_argument('--layer', type=int, default=2, help='number of layers')
parser.add_argument('--store_result', type=str, default='', help='store the result for tuning')
parser.add_argument('--force_step', type=int, default=0, help='forced step length, only used in single network')
parser.add_argument('--batch_size', type=int, default=2048, help='batch size used in training')
parser.add_argument('--copy', type=float, default=0, help='alpha of the copy module, see AAAI 21 learning from history paper')
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

max_step = data.ts_test if args.force_step == 0 else args.force_step
if not os.path.isdir('models'):
    os.mkdir('models')
save_path = 'models/' + args.data + '_' + args.network_type + str(datetime.datetime.now())[:19] + '.pkl'

ent_emb = torch.load('data/' + args.data + '/ent_emb.pt', map_location=torch.device('cuda:0'))
obj_rel_emb = torch.load('data/' + args.data + '/obj_rel_emb.pt', map_location=torch.device('cuda:0'))
sub_rel_emb = torch.load('data/' + args.data + '/sub_rel_emb.pt', map_location=torch.device('cuda:0'))

dim_in = ent_emb.shape[2] * len(args.history)

if args.network_type == 'single':
    model = FixStepAttentionModel(dim_in, args.dim, obj_rel_emb.shape[1], ent_emb.shape[1], sub_rel_emb, obj_rel_emb, dropout=args.dropout, h_att=args.attention_head, num_l=args.layer, lr=args.lr, copy=args.copy, copy_dim=ent_emb.shape[2]).cuda()
    max_mrr = 0
    max_e = 0

    for e in range(args.epoch):
        torch.cuda.empty_cache()
        print('epoch {:d}:'.format(e))
        with tqdm(total=data.ts_train + data.ts_val - args.fwd - max(args.history) - max_step) as pbar:
            # training
            train_rank_unf = list()
            for ts in range(args.fwd + max(args.history) + max_step, data.ts_train):
                hid = extract_emb(ent_emb, args.history, ts - max_step)
                if args.copy > 0:
                    batches = data.get_batches(ts, args.batch_size, require_mask=False, copy_mask_ts=max_step)
                    for b in range(len(batches[0])):
                        _, rank_unf, _ = model.step(hid, batches[0][b], batches[1][b], batches[2][b], filter_mask=None, copy_mask=batches[4][b], train=True)
                        with torch.no_grad():
                            train_rank_unf.append(rank_unf)
                else:
                    batches = data.get_batches(ts, -1, require_mask=False)
                    for b in range(len(batches[0])):
                        _, rank_unf, _ = model.step(hid, batches[0][b], batches[1][b], batches[2][b], filter_mask=None, train=True)
                        with torch.no_grad():
                            train_rank_unf.append(rank_unf)
                pbar.update(1)
            # validation
            total_rank_unf = list()
            total_rank_fil = list()
            with torch.no_grad():
                for ts in range(data.ts_train, data.ts_train + data.ts_val):
                    hid = extract_emb(ent_emb, args.history, ts - max_step)
                    if args.copy > 0:
                        batches = data.get_batches(ts, -1, require_mask=True, copy_mask_ts=max_step)
                        loss, rank_unf, rank_fil = model.step(hid, batches[0][0], batches[1][0], batches[2][0], filter_mask=batches[3][0], copy_mask=batches[4][0], train=False)
                    else:
                        batches = data.get_batches(ts, -1, require_mask=True)
                        loss, rank_unf, rank_fil = model.step(hid, batches[0][0], batches[1][0], batches[2][0], batches[3][0], train=False)
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
    model.load_state_dict(torch.load(save_path))
    rank_fil_l = list()
    with tqdm(total=data.ts_test) as pbar:
        with torch.no_grad():
            total_rank_unf = list()
            total_rank_fil = list()
            for ts in range(data.ts_train + data.ts_val, data.ts_train + data.ts_val + data.ts_test):
                hid = extract_emb(ent_emb, args.history, ts - max_step)
                if args.copy > 0:
                    batches = data.get_batches(ts, -1, require_mask=True, copy_mask_ts=max_step)
                    loss, rank_unf, rank_fil = model.step(hid, batches[0][0], batches[1][0], batches[2][0], filter_mask=batches[3][0], copy_mask=batches[4][0], train=False)
                else:
                    batches = data.get_batches(ts, -1, require_mask=True)
                    loss, rank_unf, rank_fil = model.step(hid, batches[0][0], batches[1][0], batches[2][0], batches[3][0], train=False)
                total_rank_unf.append(rank_unf.cpu())
                total_rank_fil.append(rank_fil.cpu())
                rank_fil_l.append(mrr(total_rank_fil[-1]))
                pbar.update(1)
            total_rank_unf = torch.cat(total_rank_unf)
            total_rank_fil = torch.cat(total_rank_fil)
    print(colorama.Fore.RED + 'Loading epoch {:d} with filtered MRR {:.4f}'.format(max_e, max_mrr) + colorama.Style.RESET_ALL)
    print(colorama.Fore.RED + '\traw MRR:      {:.4f} hit3: {:.4f} hit10: {:.4f}'.format(mrr(total_rank_unf), hit3(total_rank_unf), hit10(total_rank_unf)) + colorama.Style.RESET_ALL)
    print(colorama.Fore.RED + '\tfiltered MRR: {:.4f} hit3: {:.4f} hit10: {:.4f}'.format(mrr(total_rank_fil), hit3(total_rank_fil), hit10(total_rank_fil)) + colorama.Style.RESET_ALL)
    # print(colorama.Fore.RED + '\tfiltered MRR at each timestamp: '+ '\t'.join(str(float(fil)) for fil in rank_fil_l) + colorama.Style.RESET_ALL)
    if args.store_result != "":
        with open(args.store_result, encoding="utf-8", mode="a") as f:
            f.write('{:.2f}\t{:.5f}\t{:.4f}\n'.format(args.dropout, args.lr, mrr(total_rank_fil)))