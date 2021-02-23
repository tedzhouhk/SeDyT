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
        print('\tnum entity: {:d} num relation: {:d}'.format(self.num_entity, self.num_relation))
        print('\tduration train: {:d} vald: {:d} test: {:d}'.format(len(self.train_events), len(self.val_events), len(self.test_events)))

    def get_hetero_dict(self, ts):
        # get dict of events happens at time ts
        if ts >= self.ts_val:
            events = self.test_events[ts - self.ts_train - self.ts_val]
        elif ts >= self.ts_train:
            events = self.val_events[ts - self.ts_train]
        else:
            events = self.train_events[ts]
        event_dict = defaultdict(lambda: list())
        for s, r, o, _ in events:
            event_dict[('entity', 'r' + str(r), 'entity')].append((s, o))
            event_dict[('entity', '-r' + str(r), 'entity')].append((o, s))
        return event_dict


        