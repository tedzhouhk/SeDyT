import torch
import random
import numpy as np
from collections import defaultdict

class Events:
    # class that hold events in the dataset
    
    def __init__(self, dataset):
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
        self.copy_mask_ts = self.ts_train + self.ts_val + self.ts_test
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

    def get_hetero_dict(self, ts):
        # get dict of events happens at time ts
        events = self.get_events(ts)
        event_dict = defaultdict(lambda: list())
        for s, r, o, _ in events:
            event_dict[('entity', 'r' + str(r), 'entity')].append((s, o))
            event_dict[('entity', '-r' + str(r), 'entity')].append((o, s))
        return event_dict
    
    def update_copy_mask(self, ts):
        # update the history mask until timestamp ts (include ts) for the copy module
        if self.copy_mask_ts + 1 != ts:
            self.object_copy_mask_dict = dict()
            self.subject_copy_mask_dict = dict()
            for r in range(self.num_relation):
                self.object_copy_mask_dict[r] = dict()
                self.subject_copy_mask_dict[r] = dict()
                for e in range(self.num_entity):
                    self.object_copy_mask_dict[r][e] = set()
                    self.subject_copy_mask_dict[r][e] = set()
            for t in range(ts):
                events = self.get_events(t)
                for s, r, o, _ in events:
                    self.object_copy_mask_dict[r][s].add(o)
                    self.subject_copy_mask_dict[r][o].add(s)
        else:
            events = self.get_events(ts)
            for s, r, o, _ in events:
                self.object_copy_mask_dict[r][s].add(o)
                self.subject_copy_mask_dict[r][o].add(s)

    def get_batches(self, ts, bs, require_mask=True, copy_mask_ts=0):
        # get minibatches of event at time ts with batch size bs
        # copy mask contains history up to copy_mask_ts times ago
        events = self.get_events(ts)
        # adjust batchsize to average events in a batch
        if bs > 0:
            bs = round(len(events) // round(len(events) / bs))
        else:
            bs = len(events)
        subs = list()
        objs = list()
        rels = list()
        masks = list()
        copy_masks = list()
        sub = list()
        obj = list()
        rel = list()
        if require_mask:
            mask_obj_x = list()
            mask_obj_y = list()
            mask_sub_x = list()
            mask_sub_y = list()
        if copy_mask_ts > 0:
            self.update_copy_mask(ts-copy_mask_ts)
            copy_mask_obj_x = list()
            copy_mask_obj_y = list()
            copy_mask_sub_x = list()
            copy_mask_sub_y = list()
        random.shuffle(events)
        for s, r, o, _ in events:
            if len(sub) > bs:
                subs.append(torch.tensor(sub).cuda())
                objs.append(torch.tensor(obj).cuda())
                rels.append(torch.tensor(rel).cuda())
                if require_mask:
                    mask_x = torch.cat([torch.tensor(mask_sub_x), torch.tensor(mask_obj_x) + len(sub)])
                    mask_y = torch.cat([torch.tensor(mask_sub_y), torch.tensor(mask_obj_y)])
                    masks.append(torch.stack([mask_x,mask_y]).cuda())
                    mask_obj_x = list()
                    mask_obj_y = list()
                    mask_sub_x = list()
                    mask_sub_y = list()
                if copy_mask_ts > 0:
                    copy_mask_x = torch.cat([torch.tensor(copy_mask_sub_x), torch.tensor(copy_mask_obj_x) + len(sub)])
                    copy_mask_y = torch.cat([torch.tensor(copy_mask_sub_y), torch.tensor(copy_mask_obj_y)])
                    copy_masks.append(torch.stack([copy_mask_x,copy_mask_y]).cuda())
                    copy_mask_obj_x = list()
                    copy_mask_obj_y = list()
                    copy_mask_sub_x = list()
                    copy_mask_sub_y = list()
                sub = list()
                obj = list()
                rel = list()
            sub.append(s)
            obj.append(o)
            rel.append(r)
            if require_mask:
                mask_obj_y += self.object_mask_dict[r][s]
                mask_obj_x += [len(sub) - 1] * len(self.object_mask_dict[r][s])
                mask_sub_y += self.subject_mask_dict[r][o]
                mask_sub_x += [len(sub) - 1] * len(self.subject_mask_dict[r][o])
            if copy_mask_ts > 0:
                copy_mask_obj_y += list(self.object_copy_mask_dict[r][s])
                copy_mask_obj_x += [len(sub) - 1] * len(self.object_copy_mask_dict[r][s])
                copy_mask_sub_y += list(self.subject_copy_mask_dict[r][o])
                copy_mask_sub_x += [len(sub) - 1] * len(self.subject_copy_mask_dict[r][o])
        subs.append(torch.tensor(sub).cuda())
        objs.append(torch.tensor(obj).cuda())
        rels.append(torch.tensor(rel).cuda())
        if require_mask:
            mask_x = torch.cat([torch.tensor(mask_sub_x), torch.tensor(mask_obj_x) + len(sub)])
            mask_y = torch.cat([torch.tensor(mask_sub_y), torch.tensor(mask_obj_y)])
            masks.append(torch.stack([mask_x, mask_y]).cuda())
        if copy_mask_ts > 0:
            copy_mask_x = torch.cat([torch.tensor(copy_mask_sub_x), torch.tensor(copy_mask_obj_x) + len(sub)])
            copy_mask_y = torch.cat([torch.tensor(copy_mask_sub_y), torch.tensor(copy_mask_obj_y)])
            copy_masks.append(torch.stack([copy_mask_x,copy_mask_y]).cuda())
        return subs, objs, rels, masks, copy_masks
        