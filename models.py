import dgl
import torch
import numpy as np
import dgl.nn.pytorch as dglnn
import torch.nn as nn
from utils import *
from metrics import *

class Perceptron(torch.nn.Module):

    def __init__(self, in_dim, out_dim, dropout=0, norm=False, act=True):
        super(Perceptron, self).__init__()
        self.weight = torch.nn.Parameter(torch.empty(in_dim, out_dim))
        torch.nn.init.xavier_uniform_(self.weight.data)
        self.bias = torch.nn.Parameter(torch.empty(out_dim))
        torch.nn.init.zeros_(self.bias.data)
        if norm:
            self.norm = torch.nn.BatchNorm1d(out_dim, eps=1e-9, track_running_stats=True)
        self.dropout = torch.nn.Dropout(dropout)
        self.act = act

    def forward(self, f_in):
        f_in = self.dropout(f_in)
        f_in = torch.mm(f_in, self.weight) + self.bias
        if self.act:
            f_in = torch.nn.functional.relu(f_in)
            f_in = self.norm(f_in)
        return f_in

class PreTrainModel(nn.Module):
    # model for pre-training

    def __init__(self, dim_in, dim_out, dim_r, numr, nume, dropout=0, deepth=2, dim_t=0):
        super(PreTrainModel, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dim_r = dim_r
        self.numr = numr
        self.nume = nume
        self.deepth = deepth
        self.enc_time = False
        mods = dict()
        mods['entity_emb'] = nn.Embedding(nume, dim_in)
        mods['subject_relation_emb'] = nn.Embedding(numr, dim_r)
        mods['object_relation_emb'] = nn.Embedding(numr, dim_r)
        if dim_t > 0:
            mods['time_emb'] = nn.Embedding(dim_t, 1)
            self.enc_time = True
            self.dim_t = dim_t
        for l in range(self.deepth):
            # mods['norm' + str(l)] = nn.LayerNorm(dim_in)
            conv_dict = dict()
            for r in range(self.numr):
                conv_dict['r' + str(r)] = dglnn.GraphConv(dim_in, dim_out)
                conv_dict['-r' + str(r)] = dglnn.GraphConv(dim_in, dim_out)
            mods['conv' + str(l)] = dglnn.HeteroGraphConv(conv_dict,aggregate='mean')
            mods['dropout' + str(l)] = nn.Dropout(dropout)
            mods['act' + str(l)] = nn.ReLU()
            dim_in = dim_out
        mods['object_classifier'] = Perceptron(dim_out + dim_r + dim_t, nume, act=False)
        mods['subject_classifier'] = Perceptron(dim_out + dim_r + dim_t, nume, act=False)
        self.mods = nn.ModuleDict(mods)
        self.loss_fn = nn.CrossEntropyLoss(reduction='sum')

    def forward(self, sub, obj, rel, g, ts):
        h = self.mods['entity_emb'].weight
        for l in range(self.deepth):
            # h = self.mods['norm' + str(l)](h)
            h = self.mods['conv' + str(l)](g, {'entity': h})['entity']
            h = self.mods['act' + str(l)](h)
            h = self.mods['dropout' + str(l)](h)
        sub_emb = h[sub]
        obj_emb = h[obj]
        sub_rel_emb = self.mods['subject_relation_emb'](rel)
        obj_rel_emb = self.mods['object_relation_emb'](rel)
        if not self.enc_time:
            sub_predict = self.mods['subject_classifier'](torch.cat([obj_emb, obj_rel_emb], 1))
            obj_predict = self.mods['object_classifier'](torch.cat([sub_emb, sub_rel_emb], 1))
        else:
            time_emb = (self.mods['time_emb'].weight.squeeze() * ts).expand(obj_emb.shape[0], self.dim_t)
            sub_predict = self.mods['subject_classifier'](torch.cat([obj_emb, obj_rel_emb, time_emb], 1))
            obj_predict = self.mods['object_classifier'](torch.cat([sub_emb, sub_rel_emb, time_emb], 1))
        return sub_predict, obj_predict
    
    def get_entity_embedding(self, g):
        h = self.mods['entity_emb'].weight
        for l in range(self.deepth):
            h = self.mods['conv' + str(l)](g, {'entity': h})['entity']
            h = self.mods['dropout' + str(l)](h)
        return h

    def forward_and_loss(self, sub, obj, rel, g, ts, filter_mask=None):
        sub_pre, obj_pre = self.forward(sub, obj, rel, g, ts)
        pre = torch.cat([sub_pre, obj_pre])
        tru = torch.cat([sub, obj])
        loss = self.loss_fn(pre, tru)
        with torch.no_grad():
            pre = pre.clone().detach()
            tru = tru.clone().detach()
            pre_thres = pre.gather(1,tru.unsqueeze(1))
            rank_unf = get_rank(pre, pre_thres)
            rank_fil = None
            if filter_mask is not None:
                pre[filter_mask[0], filter_mask[1]] = float('-inf')
                rank_fil = get_rank(pre, pre_thres)
        return loss, rank_unf, rank_fil

class AttentionLayer(torch.nn.Module):
    
    def __init__(self, in_dim, out_dim, dropout=0, h_att=8):
        super(AttentionLayer, self).__init__()
        self.h_att = h_att
        mods = dict()
        for h in range(h_att):
            mods['dropout' + str(h)] = nn.Dropout(p=dropout)
            mods['w_v_' + str(h)] = nn.Linear(in_dim, out_dim // h_att)
        self.mods = nn.ModuleDict(mods)
        
    def forward(self, hid, adj):
        out = list()
        for h in range(self.h_att):
            hidd = self.mods['dropout' + str(h)](hid)
            v = self.mods['w_v_' + str(h)](hidd)
            out.append(torch.matmul(adj[h], v))
        out = torch.cat(out, dim=1)
        return torch.nn.functional.tanh(out)

class Attention(torch.nn.Module):

    def __init__(self, in_dim, out_dim, h_att=8):
        super(Attention, self).__init__()
        self.h_att = h_att
        mods = dict()
        for h in range(h_att):
            mods['w_q_' + str(h)] = nn.Linear(in_dim, out_dim // h_att)
            mods['w_k_' + str(h)] = nn.Linear(in_dim, out_dim // h_att)
            mods['softmax' + str(h)] = nn.Softmax(dim=1)
        self.mods = nn.ModuleDict(mods)
    
    def forward(self, hid):
        out = list()
        for h in range(self.h_att):
            q = self.mods['w_q_' + str(h)](hid)
            k = self.mods['w_q_' + str(h)](hid)
            out.append(self.mods['softmax' + str(h)](torch.matmul(q, k.T)))
        return out

class FixStepAttentionModel(torch.nn.Module):

    def __init__(self, in_dim, out_dim, dim_r, nume, sub_rel_emb, obj_rel_emb, dropout=0, h_att=8, num_l=1, lr=0.01):
        super(FixStepAttentionModel, self).__init__()
        self.num_l = num_l
        mods = dict()
        mods['subject_relation_emb'] = nn.Embedding.from_pretrained(sub_rel_emb, freeze=True)
        mods['object_relation_emb'] = nn.Embedding.from_pretrained(obj_rel_emb, freeze=True)
        mods['attention'] = Attention(in_dim, out_dim, h_att=h_att)
        for l in range(num_l):
            mods['norm_' + str(l)] = nn.LayerNorm(in_dim)
            mods['att_' + str(l)] = AttentionLayer(in_dim, out_dim, dropout=dropout, h_att=h_att)
            in_dim = out_dim
        mods['object_classifier'] = Perceptron(out_dim + dim_r, nume, act=False)
        mods['subject_classifier'] = Perceptron(out_dim + dim_r, nume, act=False)
        self.mods = nn.ModuleDict(mods)
        self.loss_fn = nn.CrossEntropyLoss(reduction='sum')
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def forward(self, hid, sub, obj, rel):
        adj = self.mods['attention'](self.mods['norm_0'](hid))
        for l in range(self.num_l):
            hid = self.mods['norm_' + str(l)](hid)
            hid = self.mods['att_' + str(l)](hid, adj)
        sub_emb = hid[sub]
        obj_emb = hid[obj]
        sub_rel_emb = self.mods['subject_relation_emb'](rel)
        obj_rel_emb = self.mods['object_relation_emb'](rel)
        sub_predict = self.mods['subject_classifier'](torch.cat([obj_emb, obj_rel_emb], 1))
        obj_predict = self.mods['object_classifier'](torch.cat([sub_emb, sub_rel_emb], 1))
        return sub_predict, obj_predict

    def step(self, hid, sub, obj, rel, filter_mask=None, train=True):
        if train:
            self.train()
            self.optimizer.zero_grad()
        else:
            self.eval()
        sub_pre, obj_pre = self.forward(hid, sub, obj, rel)
        pre = torch.cat([sub_pre, obj_pre])
        tru = torch.cat([sub, obj])
        loss = self.loss_fn(pre, tru)
        if train:
            loss.backward()
            torch.nn.utils.clip_grad_norm(self.parameters(), 5)
            self.optimizer.step()
        with torch.no_grad():
            pre = pre.clone().detach()
            tru = tru.clone().detach()
            pre_thres = pre.gather(1,tru.unsqueeze(1))
            rank_unf = get_rank(pre, pre_thres)
            rank_fil = None
            if filter_mask is not None:
                pre[filter_mask[0], filter_mask[1]] = float('-inf')
                rank_fil = get_rank(pre, pre_thres)
        return loss, rank_unf, rank_fil