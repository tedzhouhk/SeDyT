import torch
import random
import pickle
import os
import numpy as np
from collections import defaultdict
from tqdm import tqdm

class Events:
    # class that hold events in the dataset
    
    def __init__(self, dataset):
        self.f_data = dataset
        self.train_events = list()
        self.val_events = list()
        self.test_events = list()
        time_stamp = None
        first_event = True
        curr_ts = 0
        self.num_entity = 0
        self.num_relation = 0
        def update_count(s, r, o):
            if s > self.num_entity:
                self.num_entity = s
            if r > self.num_relation:
                self.num_relation = r
            if o > self.num_entity:
                self.num_entity = o
        print('loading data...')
        with open('data_raw/' + dataset + '/train.txt', 'r') as f:
            events = list()
            for l in f:
                update_count(int(l.split()[0]),int(l.split()[1]),int(l.split()[2]))
                if first_event:
                    assert l.split()[3] == '0', 'First event should start at time 0.'
                    first_event = False
                if time_stamp is None and l.split()[3] != '0':
                    time_stamp = int(l.split()[3])
                    print('\ttime duration: {:d}'.format(time_stamp))
                    self.train_events.append(events)
                    events = list()
                    curr_ts += 1
                elif time_stamp is not None and int(l.split()[3]) // time_stamp > curr_ts:
                    self.train_events.append(events)
                    events = list()
                    curr_ts += 1
                events.append([int(l.split()[0]), int(l.split()[1]), int(l.split()[2]), curr_ts])
            self.train_events.append(events)
            events = list()
            curr_ts += 1
        with open('data_raw/' + dataset + '/valid.txt', 'r') as f:
            for l in f:
                update_count(int(l.split()[0]),int(l.split()[1]),int(l.split()[2]))
                if int(l.split()[3]) // time_stamp > curr_ts:
                    self.val_events.append(events)
                    events = list()
                    curr_ts += 1
                events.append([int(l.split()[0]), int(l.split()[1]), int(l.split()[2]), curr_ts])
            self.val_events.append(events)
            events = list()
            curr_ts += 1
        with open('data_raw/' + dataset + '/test.txt', 'r') as f:
            for l in f:
                update_count(int(l.split()[0]),int(l.split()[1]),int(l.split()[2]))
                if int(l.split()[3]) // time_stamp > curr_ts:
                    self.test_events.append(events)
                    events = list()
                    curr_ts += 1
                events.append([int(l.split()[0]), int(l.split()[1]), int(l.split()[2]), curr_ts])
            self.test_events.append(events)
            events = list()
            curr_ts += 1
        self.num_entity += 1
        self.num_relation += 1
        self.ts_train = len(self.train_events)
        self.ts_val = len(self.val_events)
        self.ts_test = len(self.test_events)
        self.copy_mask_ts = 0
        print('\tnum entity: {:d} num relation: {:d}'.format(self.num_entity, self.num_relation))
        print('\tduration train: {:d} vald: {:d} test: {:d}'.format(len(self.train_events), len(self.val_events), len(self.test_events)))
        self.generate_mask_dict()
        print('Done loading data.')

    def generate_mask_dict(self):
        self.object_mask_dict = dict()
        self.subject_mask_dict = dict()
        for r in range(self.num_relation):
            self.object_mask_dict[r] = dict()
            self.subject_mask_dict[r] = dict()
            for e in range(self.num_entity):
                self.object_mask_dict[r][e] = set()
                self.subject_mask_dict[r][e] = set()
        for events in self.train_events + self.val_events + self.test_events:
            for s, r, o, _ in events:
                self.object_mask_dict[r][s].add(o)
                self.subject_mask_dict[r][o].add(s)
        for r in range(self.num_relation):
            for e in range(self.num_entity):
                self.object_mask_dict[r][e] = list(self.object_mask_dict[r][e])
                self.subject_mask_dict[r][e] = list(self.subject_mask_dict[r][e])

    def get_events(self, ts):
        if ts >= self.ts_train + self.ts_val:
            events = self.test_events[ts - self.ts_train - self.ts_val]
        elif ts >= self.ts_train:
            events = self.val_events[ts - self.ts_train]
        else:
            events = self.train_events[ts]
        return events

    def get_hetero_dict(self, ts, h):
        # get dict of events happens at time ts
        h = min(ts, h)
        events = self.get_events(ts)
        event_dict = defaultdict(lambda: list())
        offset = ts * self.num_entity
        for s, r, o, _ in events:
            event_dict[('entity', 'r' + str(r), 'entity')].append((s + offset, o + offset))
            event_dict[('entity', '-r' + str(r), 'entity')].append((o + offset, s + offset))
        for e in range(self.num_entity):
            for i in range(1, h + 1):
                h_offset = i * self.num_entity
                event_dict[('entity', 'self', 'entity')].append((e + offset - h_offset, e + offset))
        return event_dict
    
    def update_copy_mask(self, ts):
        # update the history mask until timestamp ts (include ts) for the copy module
        if self.copy_mask_ts == ts:
            pass
        elif self.copy_mask_ts + 1 != ts:
            self.object_copy_mask_dict = dict()
            self.subject_copy_mask_dict = dict()
            for r in range(self.num_relation):
                self.object_copy_mask_dict[r] = dict()
                self.subject_copy_mask_dict[r] = dict()
                for e in range(self.num_entity):
                    self.object_copy_mask_dict[r][e] = set()
                    self.subject_copy_mask_dict[r][e] = set()
            for t in range(ts + 1):
                events = self.get_events(t)
                for s, r, o, _ in events:
                    self.object_copy_mask_dict[r][s].add(o)
                    self.subject_copy_mask_dict[r][o].add(s)
        else:
            events = self.get_events(ts)
            for s, r, o, _ in events:
                self.object_copy_mask_dict[r][s].add(o)
                self.subject_copy_mask_dict[r][o].add(s)

    def generate_batches(self, copy_mask_ts=0):
        f_batch = 'data/' + self.f_data + '/copy_ts' + str(copy_mask_ts)
        self.cached_copy_mask_ts = copy_mask_ts
        if os.path.isdir(f_batch):
            print('Load batches from ' + f_batch + '...', end='', flush=True)
            with open(f_batch + '/subs.pkl', 'rb') as f:
                self.b_subs = pickle.load(f)
            with open(f_batch + '/objs.pkl', 'rb') as f:
                self.b_objs = pickle.load(f)
            with open(f_batch + '/rels.pkl', 'rb') as f:
                self.b_rels = pickle.load(f)
            with open(f_batch + '/sub_masks.pkl', 'rb') as f:
                self.b_sub_masks = pickle.load(f)
            with open(f_batch + '/obj_masks.pkl', 'rb') as f:
                self.b_obj_masks = pickle.load(f)
            with open(f_batch + '/sub_copy_masks.pkl', 'rb') as f:
                self.b_sub_copy_masks = pickle.load(f)
            with open(f_batch + '/obj_copy_masks.pkl', 'rb') as f:
                self.b_obj_copy_masks = pickle.load(f)
            print('Done')
        else:
            print('Generating batches...')
            self.b_subs = list()
            self.b_objs = list()
            self.b_rels = list()
            self.b_sub_masks = list()
            self.b_obj_masks = list()
            self.b_sub_copy_masks = list()
            self.b_obj_copy_masks = list()
            for ts in tqdm(range(self.ts_train + self.ts_val + self.ts_test)):
                subs = list()
                objs = list()
                rels = list()
                masks = list()
                copy_masks = list()
                mask_obj_x = list()
                mask_obj_y = list()
                mask_sub_x = list()
                mask_sub_y = list()
                # copy mask contains history up to copy_mask_ts times ago
                self.update_copy_mask(ts - copy_mask_ts)
                copy_mask_obj_x = list()
                copy_mask_obj_y = list()
                copy_mask_sub_x = list()
                copy_mask_sub_y = list()
                events = self.get_events(ts)
                for s, r, o, _ in events:
                    subs.append(s)
                    objs.append(o)
                    rels.append(r)
                    mask_obj_y += self.object_mask_dict[r][s]
                    mask_obj_x += [len(subs) - 1] * len(self.object_mask_dict[r][s])
                    mask_sub_y += self.subject_mask_dict[r][o]
                    mask_sub_x += [len(subs) - 1] * len(self.subject_mask_dict[r][o])
                    copy_mask_obj_y += list(self.object_copy_mask_dict[r][s])
                    copy_mask_obj_x += [len(subs) - 1] * len(self.object_copy_mask_dict[r][s])
                    copy_mask_sub_y += list(self.subject_copy_mask_dict[r][o])
                    copy_mask_sub_x += [len(subs) - 1] * len(self.subject_copy_mask_dict[r][o])
                subs = torch.tensor(subs)
                objs = torch.tensor(objs)
                rels = torch.tensor(rels)
                self.b_subs.append(subs)
                self.b_objs.append(objs)
                self.b_rels.append(rels)
                self.b_sub_masks.append(torch.zeros(subs.shape[0], self.num_entity, dtype=bool))
                self.b_obj_masks.append(torch.zeros(subs.shape[0], self.num_entity, dtype=bool))
                self.b_sub_masks[-1][torch.tensor(mask_sub_x), torch.tensor(mask_sub_y)] = 1
                self.b_obj_masks[-1][torch.tensor(mask_obj_x), torch.tensor(mask_obj_y)] = 1
                self.b_sub_copy_masks.append(torch.zeros(subs.shape[0], self.num_entity, dtype=bool))
                self.b_obj_copy_masks.append(torch.zeros(subs.shape[0], self.num_entity, dtype=bool))
                self.b_sub_copy_masks[-1][torch.tensor(copy_mask_sub_x, dtype=torch.long), torch.tensor(copy_mask_sub_y, dtype=torch.long)] = 1
                self.b_obj_copy_masks[-1][torch.tensor(copy_mask_obj_x, dtype=torch.long), torch.tensor(copy_mask_obj_y, dtype=torch.long)] = 1
            print('Saving to ' + f_batch + '...', end='', flush=True)
            os.mkdir(f_batch)
            with open(f_batch + '/subs.pkl', 'wb') as f:
                pickle.dump(self.b_subs, f)
            with open(f_batch + '/objs.pkl', 'wb') as f:
                pickle.dump(self.b_objs, f)
            with open(f_batch + '/rels.pkl', 'wb') as f:
                pickle.dump(self.b_rels, f)
            with open(f_batch + '/sub_masks.pkl', 'wb') as f:
                pickle.dump(self.b_sub_masks, f)
            with open(f_batch + '/obj_masks.pkl', 'wb') as f:
                pickle.dump(self.b_obj_masks, f)
            with open(f_batch + '/sub_copy_masks.pkl', 'wb') as f:
                pickle.dump(self.b_sub_copy_masks, f)
            with open(f_batch + '/obj_copy_masks.pkl', 'wb') as f:
                pickle.dump(self.b_obj_copy_masks, f)
            print('Done')

    def get_batches(self, ts, bs, require_mask=True, copy_mask_ts=0):
        # get minibatches of event at time ts with batch size bs
        if bs > 0:
            bs = round(self.b_subs[ts].shape[0] // round(self.b_subs[ts].shape[0] / bs))
        else:
            bs = self.b_subs[ts].shape[0]
        idx = torch.randperm(self.b_subs[ts].shape[0])
        subs = list()
        objs = list()
        rels = list()
        masks = list()
        copy_masks = list()
        start = 0
        end = bs
        while True:
            subs.append(self.b_subs[ts][idx[start:end]].cuda())
            objs.append(self.b_objs[ts][idx[start:end]].cuda())
            rels.append(self.b_rels[ts][idx[start:end]].cuda())
            masks.append(torch.cat([self.b_sub_masks[ts][idx[start:end]], self.b_obj_masks[ts][idx[start:end]]], dim=0).cuda())
            if self.cached_copy_mask_ts == copy_mask_ts:
                # use cached copy_masks
                copy_masks.append(torch.cat([self.b_sub_copy_masks[ts][idx[start:end]], self.b_obj_copy_masks[ts][idx[start:end]]], dim=0).cuda())
            else:
                # generate copy_masks with new copy_mask_ts
                self.update_copy_mask(ts - copy_mask_ts)
                new_sub_copy_masks = torch.zeros(subs[-1].shape[0], self.num_entity, dtype=bool)
                new_obj_copy_masks = torch.zeros(subs[-1].shape[0], self.num_entity, dtype=bool)
                for s, r, o, i in zip(subs[-1], rels[-1], objs[-1], range(subs[-1].shape[0])):
                    s, r, o = int(s), int(r), int(o)
                    new_sub_copy_masks[i][list(self.subject_copy_mask_dict[r][o])] = 1
                    new_obj_copy_masks[i][list(self.object_copy_mask_dict[r][s])] = 1
                copy_masks.append(torch.cat([new_sub_copy_masks, new_obj_copy_masks], dim=0).cuda())
            start = end
            if start == self.b_subs[ts].shape[0]:
                break
            end = min(end + bs, self.b_subs[ts].shape[0])
        return subs, objs, rels, masks, copy_masks
        