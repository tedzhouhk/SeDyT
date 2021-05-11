import torch
import numpy as np
import dgl
import yaml
import os
import matplotlib.pyplot as plt
from collections import defaultdict
from events import Events
from torch.utils.tensorboard import SummaryWriter

writer = None
step_dict = defaultdict(int)

def set_writer(comments):
    global writer
    if not os.path.isdir('models'):
        os.mkdir('runs')
    writer = SummaryWriter('runs/' + comments)

def get_writer():
    return writer

def get_global_step(s):
    global step_dict
    ans = step_dict[s]
    step_dict[s] += 1
    return ans

def add_virtual_relation(e, nume, numr):
    # add a virtual node that have all relationship because dgl.heterograph does not support add edge types
    for r in range(numr):
        e['entity', 'r' + str(r), 'entity'].append((nume, nume))
        e['entity', '-r' + str(r), 'entity'].append((nume, nume))
    e['entity','self','entity'].append((nume,nume))
    return
        
def add_edges_from_dict(g, event_dict):
    for etype in event_dict:
        u = list()
        v = list()
        for t in event_dict[etype]:
            u.append(t[0])
            v.append(t[1])
        g.add_edges(u, v, etype=etype)
    return

def extract_emb(ent_emb, history, ts):
    h = list()
    for t in history:
        h.append(ent_emb[ts - t])
    h = torch.cat(h, dim=1)
    return h

def plot_and_save(fil, unf):
    fix, (ax1, ax2) = plt.subplots(1, 2)
    x = np.linspace(0, 1, 101)
    ax1.plot(x, unf)
    ax1.set_title('Raw MRR')
    ax1.set_xlabel('Copy Ratio')
    ax2.plot(x, fil)
    ax2.set_title('Filtered MRR')
    ax2.set_xlabel('Copy Ratio')
    plt.savefig('copy_ratio.png', dpi=1000)

def parse_train_config(f):
    with open(f) as tf:
        tc = yaml.load(tf)
    if not 'granularity' in tc['emb-net'][0].keys():
        tc['emb-net'][0]['granularity'] = 1
    return tc['emb-net'][0], tc['gen-net'][0], tc['train'][0]