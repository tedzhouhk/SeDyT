import torch
import numpy as np
from events import Events
import dgl

def add_virtual_relation(e, nume, numr):
    # add a virtual node that have all relationship because dgl.heterograph does not support add edge types
    for r in range(numr):
        e['entity', 'r' + str(r), 'entity'].append((nume, nume))
        e['entity', '-r' + str(r), 'entity'].append((nume, nume))
    return
        
def add_edges_from_dict(g, event_dict):
    for etype in g.canonical_etypes:
        if etype in event_dict:
            u = list()
            v = list()
            for t in event_dict[etype]:
                u.append(t[0])
                v.append(t[1])
            g.add_edges(u, v, etype=etype)
    return