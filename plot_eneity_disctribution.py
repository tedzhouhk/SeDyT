from events import Events
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np


fig, axs = plt.subplots(5, 2, figsize=(15, 25))
axid = 0
for d in ['GDELT', 'ICEWS14', 'ICEWS18', 'WIKI', 'YAGO']:
    e = Events(d)
    dists = np.zeros(e.num_entity, dtype=np.int)
    disto = np.zeros(e.num_entity, dtype=np.int)
    for ev in e.train_events:
        for s, r, o, _ in ev:
            dists[s] += 1
            disto[o] += 1
    dists = np.sort(dists)[::-1]
    disto = np.sort(disto)[::-1]
    x = [i for i, _ in enumerate(dists)]
    axs[axid][0].plot(x, list(dists))
    axs[axid][1].plot(x, list(disto))
    axs[axid][0].set_title(d+' Subject')
    axs[axid][1].set_title(d+' Object')
    axid += 1

plt.savefig('subject_object_distribution.png')