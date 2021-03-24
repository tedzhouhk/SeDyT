import torch
import numpy as np
import dgl
import matplotlib.pyplot as plt
from events import Events

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