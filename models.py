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

    def __init__(self, dim_in, dim_out, dim_r, numr, nume, dropout=0, deepth=2):
        super(PreTrainModel, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dim_r = dim_r
        self.numr = numr
        self.nume = nume
        self.deepth = deepth
        mods = dict()
        mods['entity_emb'] = nn.Embedding(nume, dim_in)
        mods['subject_relation_emb'] = nn.Embedding(numr, dim_r)
        mods['object_relation_emb'] = nn.Embedding(numr, dim_r)
        for l in range(self.deepth):
            conv_dict = dict()
            for r in range(self.numr):
                conv_dict['r' + str(r)] = dglnn.GraphConv(dim_in, dim_out)
                conv_dict['-r' + str(r)] = dglnn.GraphConv(dim_in, dim_out)
            mods['conv' + str(l)] = dglnn.HeteroGraphConv(conv_dict,aggregate='mean')
            mods['dropout' + str(l)] = nn.Dropout(dropout)
            dim_in = dim_out
        mods['object_classifier'] = Perceptron(dim_out + dim_r, nume, act=False)
        mods['subject_classifier'] = Perceptron(dim_out + dim_r, nume, act=False)
        self.mods = nn.ModuleDict(mods)
        self.loss_fn = nn.CrossEntropyLoss(reduction='sum')

    def forward(self, sub, obj, rel, g):
        h = self.mods['entity_emb'].weight
        for l in range(self.deepth):
            h = self.mods['conv' + str(l)](g, {'entity': h})['entity']
            h = self.mods['dropout' + str(l)](h)
        sub_emb = h[sub]
        obj_emb = h[obj]
        sub_rel_emb = self.mods['subject_relation_emb'](rel)
        obj_rel_emb = self.mods['object_relation_emb'](rel)
        sub_predict = self.mods['subject_classifier'](torch.cat([obj_emb, obj_rel_emb], 1))
        obj_predict = self.mods['object_classifier'](torch.cat([sub_emb, sub_rel_emb], 1))
        return sub_predict, obj_predict

    def forward_and_loss(self, sub, obj, rel, g, filter_mask=None):
        sub_pre, obj_pre = self.forward(sub, obj, rel, g)
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

            

