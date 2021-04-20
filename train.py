import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, help='which dataset to use')
parser.add_argument('--config', type=str, help='configuration file')
parser.add_argument('--force_step', type=int, default=0, help='force the step length to be some number')
parser.add_argument('--gpu', type=int, default=0, help='which gpu to use')
parser.add_argument('--store_result', type=str, default='', help='store the result for tuning')
parser.add_argument('--single_step_model', type=str, default='', help='whether to load a pre-trained single-step model')
parser.add_argument('--sweep', type=str, default='', help='lr-0-0.005-25')
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
import random
import datetime
from utils import *
from models import *

data = Events(args.data)

max_step = data.ts_test + data.ts_val if args.force_step == 0 else args.force_step
if not os.path.isdir('models'):
    os.mkdir('models')
save_path = 'models/' + args.data + str(datetime.datetime.now())[:19] + '.pkl'

emb_conf, gen_conf, train_conf = parse_train_config(args.config)
set_writer(str(datetime.datetime.now())[:19].replace(' ', '') + 'LR{:.4f}DR{:.1f}'.format(train_conf['lr'], train_conf['dropout']))

max_history = max([int(h) for h in gen_conf['history'].split(' ')])

# generate graph, here we build the full graph with all nodes because a node at time t only has in-neighbors from the past
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

data.generate_batches(copy_mask_ts=max_step)

sweep_para = args.sweep.split('-')[0]
if args.sweep == '':
    # no sweep performed, just a placeholder
    sweep_para = 'lr'
    sweep_range = [train_conf[sweep_para]]
else:
    sweep_range = list(np.linspace(float(args.sweep.split('-')[1]), float(args.sweep.split('-')[2]), int(args.sweep.split('-')[3])))

for sweep_value in sweep_range:
    if sweep_para == 'fwd':
        sweep_value = int(sweep_value)
    train_conf[sweep_para] = sweep_value
    model = FixStepModel(emb_conf, gen_conf, train_conf, g, data.num_entity, data.num_relation, max_step).cuda()
    if args.single_step_model == '':
        max_mrr = 0
        max_e = 0
        for e in range(train_conf['epoch']):
            torch.cuda.empty_cache()
            print('Epoch {:d}:'.format(e))
            with tqdm(total=data.ts_train + data.ts_val - train_conf['fwd'] - max_history - max_step) as pbar:
                # training
                train_loss = 0
                train_rank_unf = list()
                for ts in range(train_conf['fwd'] + max_history + max_step, data.ts_train):
                    if gen_conf['copy'] > 0:
                        batches = data.get_batches(ts, train_conf['batch_size'], require_mask=False, copy_mask_ts=max_step)
                        for b in range(len(batches[0])):
                            ls, rank_unf, _ = model.step(batches[0][b], batches[1][b], batches[2][b], ts, filter_mask=None, copy_mask=batches[4][b], train=True)
                            with torch.no_grad():
                                train_loss += ls
                                train_rank_unf.append(rank_unf)
                    else:
                        batches = data.get_batches(ts, train_conf['batch_size'], require_mask=False)
                        for b in range(len(batches[0])):
                            ls, rank_unf, _ = model.step(batches[0][b], batches[1][b], batches[2][b], ts, filter_mask=None, train=True)
                            with torch.no_grad():
                                train_ls += ls
                                train_rank_unf.append(rank_unf)
                    pbar.update(1)
                get_writer().add_scalar('train_loss', train_loss, get_global_step('train_loss'))
                
                # validation
                valid_loss = 0
                total_rank_unf = list()
                total_rank_fil = list()
                with torch.no_grad():
                    for ts in range(data.ts_train, data.ts_train + data.ts_val):
                        rank_unf_e = list()
                        rank_fil_e = list()
                        if gen_conf['copy'] > 0:
                            batches = data.get_batches(ts, train_conf['batch_size'], require_mask=True, copy_mask_ts=max_step)
                            for b in range(len(batches[0])):
                                loss, rank_unf, rank_fil = model.step(batches[0][b], batches[1][b], batches[2][b], ts, filter_mask=batches[3][b], copy_mask=batches[4][b], train=False)
                                valid_loss += loss
                                rank_unf_e.append(rank_unf)
                                rank_fil_e.append(rank_fil)
                        else:
                            batches = data.get_batches(ts, train_conf['batch_size'], require_mask=True)
                            for b in range(len(batches[0])):
                                loss, rank_unf, rank_fil = model.step(batches[0][0], batches[1][0], batches[2][0], ts, filter_mask=batches[3][0], train=False)
                                valid_loss += loss
                                rank_unf_e.append(rank_unf)
                                rank_fil_e.append(rank_fil)
                        rank_unf_e = torch.cat(rank_unf_e)
                        rank_fil_e = torch.cat(rank_fil_e)
                        total_rank_unf.append(rank_unf_e)
                        total_rank_fil.append(rank_fil_e)
                        total_rank_unf.append(rank_unf)
                        total_rank_fil.append(rank_fil)
                        pbar.update(1)
                    total_rank_unf = torch.cat(total_rank_unf)
                    total_rank_fil = torch.cat(total_rank_fil)
                train_rank_unf = torch.cat(train_rank_unf)
            get_writer().add_scalar('valid_loss', valid_loss, get_global_step('valid_loss'))
            get_writer().add_scalar('train_raw_MRR', mrr(train_rank_unf), get_global_step('train_raw_MRR'))
            get_writer().add_scalar('valid_raw_MRR', mrr(total_rank_unf), get_global_step('valid_raw_MRR'))
            get_writer().add_scalar('valid_fil_MRR', mrr(total_rank_fil), get_global_step('valid_fil_MRR'))
            print('\ttrain raw MRR:      {:.4f}'.format(mrr(train_rank_unf)))
            print('\tvalid raw MRR:      {:.4f} hit3: {:.4f} hit10: {:.4f}'.format(mrr(total_rank_unf), hit3(total_rank_unf), hit10(total_rank_unf)))
            print('\tvalid filtered MRR: {:.4f} hit3: {:.4f} hit10: {:.4f}'.format(mrr(total_rank_fil), hit3(total_rank_fil), hit10(total_rank_fil)))
            if mrr(total_rank_fil) > max_mrr:
                max_mrr = mrr(total_rank_fil)
                max_e = e
                torch.save(model.state_dict(), save_path)

            if len(sweep_range) == 1:
                # show test result every epoch for quicker debugging
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
                                    loss, rank_unf, rank_fil = model.step(batches[0][b], batches[1][b], batches[2][b], ts, filter_mask=batches[3][b], copy_mask=batches[4][b], train=False)
                                    rank_unf_e.append(rank_unf)
                                    rank_fil_e.append(rank_fil)
                            else:
                                batches = data.get_batches(ts, train_conf['batch_size'], require_mask=True)
                                for b in range(len(batches[0])):
                                    loss, rank_unf, rank_fil = model.step(batches[0][b], batches[1][b], batches[2][b], ts, filter_mask=batches[3][b], train=False)
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
                get_writer().add_scalar('test_raw_MRR', mrr(total_rank_unf),get_global_step('test_raw_MRR'))
                get_writer().add_scalar('test_fil_MRR', mrr(total_rank_fil),get_global_step('test_fil_MRR'))
                print(colorama.Fore.RED + '\traw MRR:      {:.4f} hit3: {:.4f} hit10: {:.4f}'.format(mrr(total_rank_unf), hit3(total_rank_unf), hit10(total_rank_unf)) + colorama.Style.RESET_ALL)
                print(colorama.Fore.RED + '\tfiltered MRR: {:.4f} hit3: {:.4f} hit10: {:.4f}'.format(mrr(total_rank_fil), hit3(total_rank_fil), hit10(total_rank_fil)) + colorama.Style.RESET_ALL)
    else:
        # load pre-trained model
        original_state_dict = torch.load(args.single_step_model)
        truncated_state_dict = dict()
        for k, v in original_state_dict.items():
            k = k.replace('_src','')
            k = k.replace('_dst','')
            truncated_state_dict[k] = v
        model.load_state_dict(state_dict=truncated_state_dict)
        max_e = 0

    # testing
    print(colorama.Fore.RED + 'Testing...'+ colorama.Style.RESET_ALL)
    if max_e > 0:
        print(colorama.Fore.RED + 'Loading epoch {:d} with filtered MRR {:.4f}'.format(max_e, max_mrr) + colorama.Style.RESET_ALL)
        model.load_state_dict(torch.load(save_path))
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
                        loss, rank_unf, rank_fil = model.step(batches[0][b], batches[1][b], batches[2][b], ts, filter_mask=batches[3][b], copy_mask=batches[4][b], train=False)
                        rank_unf_e.append(rank_unf)
                        rank_fil_e.append(rank_fil)
                else:
                    batches = data.get_batches(ts, train_conf['batch_size'], require_mask=True)
                    for b in range(len(batches[0])):
                        loss, rank_unf, rank_fil = model.step(batches[0][b], batches[1][b], batches[2][b], ts, filter_mask=batches[3][b], train=False)
                        rank_unf_e.append(rank_unf)
                        rank_fil_e.append(rank_fil)
                rank_unf_e = torch.cat(rank_unf_e)
                rank_fil_e = torch.cat(rank_fil_e)
                total_rank_unf.append(rank_unf_e.cpu())
                total_rank_fil.append(rank_fil_e.cpu())
                print(mrr(total_rank_fil[-1]))
                pbar.update(1)
                last_rank_unf = total_rank_unf[-1]
                last_rank_fil = total_rank_fil[-1]
            total_rank_unf = torch.cat(total_rank_unf)
            total_rank_fil = torch.cat(total_rank_fil)
    if args.single_step_model != '':
        torch.save(model.state_dict(), save_path)
    print(colorama.Fore.RED + '\traw MRR:      {:.4f} hit3: {:.4f} hit10: {:.4f}'.format(mrr(total_rank_unf), hit3(total_rank_unf), hit10(total_rank_unf)) + colorama.Style.RESET_ALL)
    print(colorama.Fore.RED + '\tfiltered MRR: {:.4f} hit3: {:.4f} hit10: {:.4f}'.format(mrr(total_rank_fil), hit3(total_rank_fil), hit10(total_rank_fil)) + colorama.Style.RESET_ALL)
    if args.store_result != "":
        with open(args.store_result, encoding="utf-8", mode="a") as f:
            f.write('single\t{:.4f}\t{:.4f}\n'.format(sweep_value, mrr(total_rank_fil)))

    if not args.force_step > 0:   
        # multi-step fine-tuning
        print('Fine-tuning for multi-step models...')
        # the result of the last timestamp is the same as single model
        ms_total_rank_unf = [last_rank_unf.cpu()]
        ms_total_rank_fil = [last_rank_fil.cpu()]
        for tts in range(data.ts_train + data.ts_val, data.ts_train + data.ts_val + data.ts_test - 1):
            model.load_state_dict(torch.load(save_path))
            step = tts - data.ts_train + 1
            print('Timestamp {:d} with step {:d}:'.format(tts, step))
            model.inf_step = step
            # training
            train_loss = 0
            train_rank_unf = list()
            for ts in tqdm(range(train_conf['fwd'] + max_history + step, data.ts_train)):
                if gen_conf['copy'] > 0:
                    batches = data.get_batches(ts, train_conf['batch_size'], require_mask=False, copy_mask_ts=step)
                    for b in range(len(batches[0])):
                        ls, rank_unf, _ = model.step(batches[0][b], batches[1][b], batches[2][b], ts, filter_mask=None, copy_mask=batches[4][b], train=True, log=False, freeze_emb=True)
                        with torch.no_grad():
                            train_loss += ls
                            train_rank_unf.append(rank_unf)
                else:
                    batches = data.get_batches(ts, train_conf['batch_size'], require_mask=False)
                    for b in range(len(batches[0])):
                        ls, rank_unf, _ = model.step(batches[0][b], batches[1][b], batches[2][b], ts, filter_mask=None, train=True, log=False, freeze_emb=True)
                        with torch.no_grad():
                            train_ls += ls
                            train_rank_unf.append(rank_unf)
            train_rank_unf = torch.cat(train_rank_unf)
            # test on the selecting ts (tts)
            with torch.no_grad():
                ts = tts
                rank_unf_e = list()
                rank_fil_e = list()
                if gen_conf['copy'] > 0:
                    batches = data.get_batches(ts, train_conf['batch_size'], require_mask=True, copy_mask_ts=step)
                    for b in range(len(batches[0])):
                        loss, rank_unf, rank_fil = model.step(batches[0][b], batches[1][b], batches[2][b], ts, filter_mask=batches[3][b], copy_mask=batches[4][b], train=False, log=False)
                        rank_unf_e.append(rank_unf)
                        rank_fil_e.append(rank_fil)
                else:
                    batches = data.get_batches(ts, train_conf['batch_size'], require_mask=True)
                    for b in range(len(batches[0])):
                        loss, rank_unf, rank_fil = model.step(batches[0][b], batches[1][b], batches[2][b], ts, filter_mask=batches[3][b], train=False, log=False)
                        rank_unf_e.append(rank_unf)
                        rank_fil_e.append(rank_fil)
                rank_unf_e = torch.cat(rank_unf_e)
                rank_fil_e = torch.cat(rank_fil_e)
                print('\ttrain raw MRR:      {:.4f}'.format(mrr(train_rank_unf)))
                print('\ttest raw MRR:       {:.4f} hit3: {:.4f} hit10: {:.4f}'.format(mrr(rank_unf_e), hit3(rank_unf_e), hit10(rank_unf_e)))
                print('\ttest filtered MRR:  {:.4f} hit3: {:.4f} hit10: {:.4f}'.format(mrr(rank_fil_e), hit3(rank_fil_e), hit10(rank_fil_e)))
                ms_total_rank_unf.append(rank_unf_e.cpu())
                ms_total_rank_fil.append(rank_fil_e.cpu())
        ms_total_rank_unf = torch.cat(ms_total_rank_unf)
        ms_total_rank_fil = torch.cat(ms_total_rank_fil)
        print(colorama.Fore.RED + 'Testing (Multi-Step)...'+ colorama.Style.RESET_ALL)
        print(colorama.Fore.RED + '\traw MRR:      {:.4f} hit3: {:.4f} hit10: {:.4f}'.format(mrr(ms_total_rank_unf), hit3(ms_total_rank_unf), hit10(ms_total_rank_unf)) + colorama.Style.RESET_ALL)
        print(colorama.Fore.RED + '\tfiltered MRR: {:.4f} hit3: {:.4f} hit10: {:.4f}'.format(mrr(ms_total_rank_fil), hit3(ms_total_rank_fil), hit10(ms_total_rank_fil)) + colorama.Style.RESET_ALL)
        if args.store_result != "":
            with open(args.store_result, encoding="utf-8", mode="a") as f:
                f.write('multi\t{:.4f}\t{:.4f}\n'.format(sweep_value, mrr(ms_total_rank_fil)))
    
    print(colorama.Fore.RED + '\tmaximum GPU memory used: {:d}MB'.format(torch.cuda.max_memory_reserved() // 1024 // 1024) + colorama.Style.RESET_ALL)