from events import Events
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np


fig, axs = plt.subplots(5, 3, figsize=(20, 25))
axid = 0
for d in ['GDELT', 'ICEWS14', 'ICEWS18', 'WIKI', 'YAGO']:
    e = Events(d)
    dists = np.zeros(e.num_entity, dtype=np.int)
    disto = np.zeros(e.num_entity, dtype=np.int)
    distr = np.zeros(e.num_relation, dtype=np.int)
    for ev in e.train_events:
        for s, r, o, _ in ev:
            dists[s] += 1
            disto[o] += 1
            distr[r] += 1
    dists = np.sort(dists)[::-1]
    disto = np.sort(disto)[::-1]
    distr = np.sort(distr)[::-1]
    distr_remain = np.cumsum(distr).astype(np.float)
    distr_remain /= np.max(distr_remain) / np.max(distr)
    x = [i for i, _ in enumerate(dists)]
    xr = [i for i, _ in enumerate(distr)]
    axs[axid][0].plot(x, list(dists))
    axs[axid][1].plot(x, list(disto))
    axs[axid][2].plot(xr, list(distr))
    axs[axid][2].plot(xr, list(distr_remain))
    axs[axid][0].set_title(d+' Subject')
    axs[axid][1].set_title(d+' Object')
    axs[axid][2].set_title(d+' Relation')
    axid += 1

plt.savefig('distribution.png')