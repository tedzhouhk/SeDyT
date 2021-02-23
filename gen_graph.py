import os
import torch
import pickle
import numpy as np
import scipy.sparse as sp
from tqdm import tqdm
from collections import Counter

def to_normed_torch_sp(i0, i1, i2, cnt, nr, ne):
    v = list()
    for r, s, o in zip(i0, i1, i2):
        v.append(1 / (cnt[str(r) + str(o)]))
    i = torch.LongTensor([i0, i1, i2])
    v = torch.FloatTensor(v)
    return torch.sparse.FloatTensor(i, v, torch.Size([nr, ne, ne]))

if not os.path.isdir('data'):
    os.mkdir('data')
dataset = ['GDELT', 'ICEWS14', 'ICEWS18', 'WIKI', 'YAGO']
tss = [15, 24, 24, 1, 1]
numr = [240, 260, 256, 24, 10]
nume = [7691, 12498, 23033, 12554, 10623]

for d, ts, nr, ne in zip(dataset, tss, numr, nume):
    print('processing ', d)
    if not os.path.isdir('data/' + d):
        os.mkdir('data/' + d)
    # [subject, relation, object, time]
    train_event = list()
    with open('data_raw/' + d + '/train.txt', 'r') as f:
        for l in f:
            train_event.append([int(l.split()[0]), int(l.split()[1]), int(l.split()[2]), int(l.split()[3])])
    val_event = list()
    with open('data_raw/' + d + '/valid.txt', 'r') as f:
        for l in f:
            val_event.append([int(l.split()[0]), int(l.split()[1]), int(l.split()[2]), int(l.split()[3])])
    test_event = list()
    with open('data_raw/' + d + '/test.txt', 'r') as f:
        for l in f:
            test_event.append([int(l.split()[0]), int(l.split()[1]), int(l.split()[2]), int(l.split()[3])])
    # delta t = duration of test events
    dt = (test_event[-1][3] - val_event[-1][3]) // ts
    print('    deltat = ', dt)
    # here we fast-forward the event of in the first delta t time period
    # pre-train task: predicting relations happened between two nodes
    assert (dt * ts < train_event[-1][3]), 'Training data too short'
    adjs = list()
    adjs_t = list()
    # i0-i2: 3 dim of sparse heterogeneous adj
    i0 = list() # i0: dim of shape |r|
    i1 = list() # i1: dim of shape |e|
    i2 = list() # i2: dim of shape |e|
    cnt = Counter()
    # add fast fowarded event:
    event_idx = 0
    while train_event[event_idx][3] < dt * ts:
        event = train_event[event_idx]
        i0.append(event[1])
        i1.append(event[0])
        i2.append(event[2])
        cnt[str(event[1]) + str(event[2])] += 1
        event_idx += 1
    t = train_event[event_idx][3]
    for event in tqdm(train_event[event_idx:] + val_event):
        if event[3] != t:
            adjs.append(to_normed_torch_sp(i0, i1, i2, cnt, nr, ne))
            adjs_t.append(t // ts)
            t = event[3]
        i0.append(event[1])
        i1.append(event[0])
        i2.append(event[2])
        cnt[str(event[1]) + str(event[2])] += 1
    adjs.append(to_normed_torch_sp(i0, i1, i2, cnt, nr, ne))
    adjs_t.append(val_event[-1][3] // ts)
    with open('data/' + d + '/adjs.pickle', 'wb') as h:
        pickle.dump(adjs, h)
    with open('data/' + d + '/adjs_t.pickle', 'wb') as h:
        pickle.dump(adjs_t, h)
    os.system('cp data_raw/' + d + '/*.txt data/' + d + '/')