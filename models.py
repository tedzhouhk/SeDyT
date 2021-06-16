import dgl
import torch
import math
import time
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
        self.norm = norm
        if norm:
            self.norm = torch.nn.BatchNorm1d(out_dim, eps=1e-9, track_running_stats=True)
        self.dropout = torch.nn.Dropout(dropout)
        self.act = act

    def forward(self, f_in):
        f_in = self.dropout(f_in)
        f_in = torch.mm(f_in, self.weight) + self.bias
        if self.act:
            f_in = torch.nn.functional.relu(f_in)
        if self.norm:
            f_in = self.norm(f_in)
        return f_in

    def reset_parameters():
        torch.nn.init.xavier_uniform_(self.weight.data)
        torch.nn.init.zeros_(self.bias.data)

class TimeEnc(nn.Module):
    # generate time encoding (TGAT, ICLR'21) from node number (nid)

    def __init__(self, dim_t, nume):
        super(TimeEnc, self).__init__()
        self.dim_t = dim_t
        self.nume = nume
        self.basis_freq = torch.nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 9, self.dim_t))).float())
        self.phase = torch.nn.Parameter(torch.zeros(self.dim_t).float())

    def forward(self, nid, ts):
        t = ts - (nid // self.nume)
        t = t.view(-1, 1) * self.basis_freq + self.phase
        return torch.cos(t)

class EmbModule(nn.Module):

    def __init__(self, dim_in, dim_out, dim_t, numr, nume, g, dropout=0, deepth=2, sampling=None, granularity=1, r_limit=None):
        super(EmbModule, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dim_t = dim_t
        self.numr = numr
        self.nume = nume
        self.deepth = deepth
        self.g = g
        self.granularity = granularity
        mods = dict()
        mods['time_enc'] = TimeEnc(dim_t, nume)
        mods['entity_emb'] = nn.Embedding(nume, dim_in)
        if r_limit is None:
            r_limit = numr
        for l in range(self.deepth):
            mods['norm' + str(l)] = nn.LayerNorm(dim_in + dim_t)
            # mods['dropout' + str(l)] = nn.Dropout(dropout)
            conv_dict = dict()
            for r in range(r_limit):
                conv_dict['r' + str(r)] = dglnn.GATConv(dim_in + dim_t, dim_out // 4, 4, feat_drop=dropout, attn_drop=dropout, residual=False)
                conv_dict['-r' + str(r)] = dglnn.GATConv(dim_in + dim_t, dim_out // 4, 4, feat_drop=dropout, attn_drop=dropout, residual=False)
                conv_dict['self'] = dglnn.GATConv(dim_in + dim_t, dim_out // 4, 4, feat_drop=dropout, attn_drop=dropout, residual=False)
                # conv_dict['r' + str(r)] = dglnn.GraphConv(dim_in + dim_t, dim_out)
                # conv_dict['-r' + str(r)] = dglnn.GraphConv(dim_in + dim_t, dim_out)
                # conv_dict['self'] = dglnn.GraphConv(dim_in + dim_t, dim_out)
            mods['conv' + str(l)] = dglnn.HeteroGraphConv(conv_dict, aggregate='mean')
            mods['act' + str(l)] = nn.ReLU()
            dim_in = dim_out
        self.mods = nn.ModuleDict(mods)
        if sampling is not None:
            fanouts = [int(d) for d in sampling.split('/')]
            self.sampler = dgl.dataloading.MultiLayerNeighborSampler(fanouts = fanouts)
        else:
            self.sampler = dgl.dataloading.MultiLayerFullNeighborSampler(self.deepth)

    def forward(self, ent, hist_ts, ts, log=True, phi_offset=0):
        tss = time.time()
        offset = (hist_ts // self.granularity) * self.nume
        ent = ent.repeat_interleave(offset.shape[0]).view(ent.shape[0], -1).cpu()
        root = torch.flatten(ent + offset)
        # return self.mods['entity_emb'](torch.remainder(root.cuda(), self.nume))
        # dgl sampler need input to be unique
        root, root_idx = torch.unique(root, sorted=True, return_inverse=True)
        blocks = self.sampler.sample_blocks(self.g, root)
        blk = [blk.to('cuda:0') for blk in blocks]
        if log:
            get_writer().add_scalar('time_sampling', time.time() - tss, get_global_step('time_sampling'))
        tss = time.time()
        # print(root.shape[0], blk[0].srcdata['_ID'].shape[0])
        h = self.mods['entity_emb'](torch.remainder(blk[0].srcdata['_ID'], self.nume))
        for l in range(self.deepth):
            phi = self.mods['time_enc'](blk[l].srcdata['_ID'], ts + phi_offset)
            h = torch.cat([h, phi], dim=1)
            h = self.mods['norm' + str(l)](h)
            # h = self.mods['dropout' + str(l)](h)
            h = self.mods['conv' + str(l)](blk[l], {'entity': h})['entity']
            h = h.view(h.shape[0], -1)
            h = self.mods['act' + str(l)](h)
        h = h[root_idx].view(-1, offset.shape[0], h.shape[-1])
        if log:
            get_writer().add_scalar('time_emb', time.time() - tss, get_global_step('time_emb'))
        return h.view(h.shape[0], -1)
    
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
        return torch.nn.functional.relu(out)

class Copy(torch.nn.Module):
    # copy module used in AAAI'21 CyGNet

    def __init__(self, in_dim, dim_r, nume, numr, dropout=0):
        super(Copy, self).__init__()
        mods = dict()
        mods['subject_relation_emb'] = nn.Embedding(numr, dim_r)
        mods['object_relation_emb'] = nn.Embedding(numr, dim_r)
        mods['object_classifier'] = Perceptron(in_dim + dim_r, nume, act=False, dropout=dropout)
        mods['subject_classifier'] = Perceptron(in_dim + dim_r, nume, act=False, dropout=dropout)
        self.mods = nn.ModuleDict(mods)
    
    def forward(self, sub_emb, obj_emb, rel, copy_mask):
        sub_rel_emb = self.mods['subject_relation_emb'](rel)
        obj_rel_emb = self.mods['object_relation_emb'](rel)
        raw_sub_predict = self.mods['subject_classifier'](torch.cat([obj_emb, obj_rel_emb], 1))
        raw_obj_predict = self.mods['object_classifier'](torch.cat([sub_emb, sub_rel_emb], 1))
        masked_predict = torch.tensor([-100.0]).cuda().repeat(raw_sub_predict.shape[0] * 2, raw_sub_predict.shape[1])
        raw_predict = torch.cat([raw_sub_predict, raw_obj_predict], dim=0)
        masked_predict[copy_mask] = raw_predict[copy_mask]
        return masked_predict[:masked_predict.shape[0] // 2], masked_predict[masked_predict.shape[0] // 2:]

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
        # trick from Transformer paper: to avoid gradient vanishing.
        # var_norm = math.sqrt(self.mods['w_k_0'].weight.shape[-1])
        for h in range(self.h_att):
            q = self.mods['w_q_' + str(h)](hid)
            k = self.mods['w_k_' + str(h)](hid)
            a = torch.nn.functional.leaky_relu(torch.matmul(q, torch.transpose(k, -1, -2)), negative_slope=0.1)
            out.append(self.mods['softmax' + str(h)](a))
            # out.append(self.mods['softmax' + str(h)](torch.matmul(q, torch.transpose(k, -1, -2)) / var_norm))
        return out

class SelfAttention(torch.nn.Module):
    
    def __init__(self, in_dim, hist_l, emb_dim, h_att=8, dropout=0):
        super(SelfAttention, self).__init__()
        self.emb_dim = emb_dim
        self.in_dim = in_dim
        self.h_att = h_att
        self.hist_l = hist_l
        mods = dict()
        mods['attention'] = Attention(in_dim // hist_l, emb_dim * h_att, h_att=h_att)
        for h in range(h_att):
            mods['dropout' + str(h)] = nn.Dropout(p=dropout)
            mods['w_v_' + str(h)] = nn.Linear(in_dim // hist_l, emb_dim // h_att)
        self.mods = nn.ModuleDict(mods)

    def forward(self, hid):
        hid = hid.view(hid.shape[0], self.hist_l, -1)
        att = self.mods['attention'](hid)
        ans = list()
        for h in range(self.h_att):
            hidd = self.mods['dropout' + str(h)](hid)
            v = self.mods['w_v_' + str(h)](hidd)
            ans.append(torch.matmul(att[h], v))
        ans = torch.cat(ans, dim=-1)
        # import pdb; pdb.set_trace()
        return torch.nn.functional.relu(ans).view(ans.shape[0], -1)

class Conv(torch.nn.Module):

    def __init__(self, in_dim, emb_dim, dropout=0):
        super(Conv, self).__init__()
        self.emb_dim = emb_dim
        mods = dict()
        mods['conv'] = torch.nn.Conv2d(1, 1, kernel_size=3, stride=1)
        mods['dropout'] = torch.nn.Dropout(dropout)
        self.mods = nn.ModuleDict(mods)

    def forward(self, hid):
        hid = hid.view(hid.shape[0], 1, -1, self.emb_dim)
        hid = self.mods['dropout'](hid)
        hid = self.mods['conv'](hid)
        hid = hid.view(hid.shape[0], -1)
        return torch.nn.functional.relu(hid)

class RNN(nn.Module):

    def __init__(self, in_dim, emb_dim, out_dim, dropout=0):
        super(RNN, self).__init__()
        self.emb_dim = emb_dim
        mods = dict()
        mods['rnn'] = torch.nn.RNN(emb_dim, hidden_size=out_dim, num_layers=2, batch_first=True, dropout=dropout)
        self.mods = nn.ModuleDict(mods)

    def forward(self, hid):
        hid = hid.view(hid.shape[0], -1, self.emb_dim)
        hid = self.mods['rnn'](hid)
        return hid[0][:,-1,:]

class LSTM(nn.Module):

    def __init__(self, in_dim, emb_dim, out_dim, dropout=0):
        super(LSTM, self).__init__()
        self.emb_dim = emb_dim
        mods = dict()
        mods['lstm'] = torch.nn.LSTM(emb_dim, hidden_size=out_dim, num_layers=2, batch_first=True, dropout=dropout)
        self.mods = nn.ModuleDict(mods)

    def forward(self, hid):
        hid = hid.view(hid.shape[0], -1, self.emb_dim)
        hid = self.mods['lstm'](hid)
        return hid[0][:,-1,:]

class GRU(nn.Module):

    def __init__(self, in_dim, emb_dim, out_dim, dropout=0):
        super(GRU, self).__init__()
        self.emb_dim = emb_dim
        mods = dict()
        mods['gru'] = torch.nn.GRU(emb_dim, hidden_size=out_dim, num_layers=2, batch_first=True, dropout=dropout)
        self.mods = nn.ModuleDict(mods)

    def forward(self, hid):
        hid = hid.view(hid.shape[0], -1, self.emb_dim)
        hid = self.mods['gru'](hid)
        return hid[0][:,-1,:]

class FixStepModel(torch.nn.Module):

    def __init__(self, emb_conf, gen_conf, train_conf, g, nume, numr, step, s_dist=None, o_dist=None):
        super(FixStepModel, self).__init__()
        # self.copy = gen_conf['copy']
        self.emb_dim = emb_conf['dim']
        self.gen_dim = [int(d) for d in gen_conf['dim'].split('-')]
        self.gen_arch = gen_conf['arch'].split('-')
        self.gen_att_h = [int(h) for h in gen_conf['att_head'].split('-')]
        self.gen_l = len(self.gen_dim)
        self.gen_hist = torch.tensor([int(x) for x in gen_conf['history'].split()]).cpu()
        self.inf_step = step
        self.train_conf = train_conf
        self.norm_loss = False
        if 'norm_loss' in train_conf:
            if train_conf['norm_loss']:
                self.norm_loss = True
                self.s_dist = torch.from_numpy(s_dist).cuda()
                self.o_dist = torch.from_numpy(o_dist).cuda()
        mods = dict()
        self.time_emb = False
        if 'dim_t' in gen_conf:
            self.time_emb = True
            self.time_emb_vec = torch.nn.Parameter(torch.Tensor(1, gen_conf['dim_t']), requires_grad=False)
            torch.nn.init.xavier_uniform_(self.time_emb_vec, gain=torch.nn.init.calculate_gain('relu'))
        r_limit = None if not 'r_limit' in emb_conf else emb_conf['r_limit']
        mods['emb'] = EmbModule(emb_conf['dim_e'], emb_conf['dim'], emb_conf['dim_t'], numr, nume, g, train_conf['dropout'], emb_conf['layer'], sampling=emb_conf['sample'], granularity=emb_conf['granularity'], r_limit=r_limit)
        # if self.copy > 0:
        mods['copy'] = Copy(self.emb_dim, gen_conf['dim_r'], nume, numr, dropout=train_conf['dropout'])
        mods['subject_relation_emb'] = nn.Embedding(numr, gen_conf['dim_r'])
        mods['object_relation_emb'] = nn.Embedding(numr, gen_conf['dim_r'])
        in_dim = emb_conf['dim'] * self.gen_hist.shape[0]
        for arch, out_dim, att_h, l in zip(self.gen_arch, self.gen_dim, self.gen_att_h, list(range(self.gen_l))):
            mods['norm_' + str(l)] = nn.LayerNorm(in_dim)
            if arch == 'dense':
                mods['layer_' + str(l)] = Perceptron(in_dim, out_dim, dropout=train_conf['dropout'])
            elif arch == 'selfatt':
                mods['layer_' + str(l)] = SelfAttention(in_dim, self.gen_hist.shape[0], out_dim // self.gen_hist.shape[0], att_h, dropout=train_conf['dropout'])
            elif arch == 'conv':
                mods['layer_' + str(l)] = Conv(in_dim, emb_conf['dim'], dropout=train_conf['dropout'])
            elif arch == 'rnn':
                mods['layer_' + str(l)] = RNN(in_dim, emb_conf['dim'], out_dim, dropout=train_conf['dropout'])
            elif arch == 'lstm':
                mods['layer_' + str(l)] = LSTM(in_dim, emb_conf['dim'], out_dim, dropout=train_conf['dropout'])
            elif arch == 'gru':
                mods['layer_' + str(l)] = GRU(in_dim, emb_conf['dim'], out_dim, dropout=train_conf['dropout'])
            else:
                raise NotImplementedError
            in_dim = out_dim
        rediual_dim = sum(self.gen_dim)
        dim_t = 0 if not self.time_emb else gen_conf['dim_t']
        mods['object_classifier'] = Perceptron(rediual_dim + gen_conf['dim_r'] + dim_t, nume, act=False, dropout=train_conf['dropout'])
        mods['subject_classifier'] = Perceptron(rediual_dim + gen_conf['dim_r'] + dim_t, nume, act=False, dropout=train_conf['dropout'])
        self.mods = nn.ModuleDict(mods)
        self.loss_fn = nn.CrossEntropyLoss(reduction='none')
        self.copy_loss_fn = nn.CrossEntropyLoss(reduction='none')
        self.optimizer = torch.optim.Adam(self.parameters(), lr=train_conf['lr'], weight_decay=train_conf['weight_decay'], amsgrad=False)
        self.copy_optimizer = torch.optim.Adam(self.mods['copy'].parameters(), lr=train_conf['lr'], weight_decay=train_conf['weight_decay'], amsgrad=False)
        self._deb=0
    
    def reset_gen_parameters(self):
        # reset parameters for generation network and optimizers
        for l in range(self.gen_l):
            for m in [self.mods['norm_' + str(l)]]:
                if hasattr(m, 'reset_parameters'):
                    m.reset_parameters()
            for m in self.mods['layer_' + str(l)].children():
                if hasattr(m, 'reset_parameters'):
                    m.reset_parameters()
        for m in self.mods['copy'].children():
            if hasattr(m, 'reset_parameters'):
                m.reset_parameters()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.train_conf['lr'], weight_decay=self.train_conf['weight_decay'], amsgrad=False)
        self.copy_optimizer = torch.optim.Adam(self.mods['copy'].parameters(), lr=self.train_conf['lr'], weight_decay=self.train_conf['weight_decay'], amsgrad=False)

    def forward(self, sub, obj, rel, ts, copy_mask=None, freeze_emb=False, log=True, phi_offset=0):
        hid = self.mods['emb'](torch.cat([sub, obj]), ts - self.inf_step - self.gen_hist, ts, log=log, phi_offset=phi_offset)
        if freeze_emb:
            hid = hid.detach()
        tss = time.time()
        copy_sub_predict = None
        copy_obj_predict = None
        # if self.copy > 0:
        copy_hid = hid[:, -self.emb_dim:]
        # only propagate gradients within the copy module
        copy_sub_predict, copy_obj_predict = self.mods['copy'](copy_hid[:sub.shape[0]].detach(), copy_hid[sub.shape[0]:].detach(), rel, copy_mask)
        # if self._deb > 100:
        #     _emb=hid
        #     import pdb; pdb.set_trace()
        residual_hid = list()
        for l in range(self.gen_l):
            hid = self.mods['norm_' + str(l)](hid)
            hid = self.mods['layer_' + str(l)](hid)
            residual_hid.append(hid)
        hid = torch.cat(residual_hid, dim=1)
        sub_emb = hid[:sub.shape[0]]
        obj_emb = hid[sub.shape[0]:]
        sub_rel_emb = self.mods['subject_relation_emb'](rel)
        obj_rel_emb = self.mods['object_relation_emb'](rel)
        if self.time_emb:
            t_emb = self.time_emb_vec[0]
            t_emb = (t_emb * ts).repeat(sub_emb.shape[0]).reshape(-1, t_emb.shape[0])
            sub_emb = torch.cat([sub_emb, t_emb], 1)
            obj_emb = torch.cat([obj_emb, t_emb], 1)
        sub_predict = self.mods['subject_classifier'](torch.cat([obj_emb, obj_rel_emb], 1))
        obj_predict = self.mods['object_classifier'](torch.cat([sub_emb, sub_rel_emb], 1))
        if log:
            get_writer().add_scalar('time_gen', time.time() - tss, get_global_step('time_gen'))
        return sub_predict, obj_predict, copy_sub_predict, copy_obj_predict

    def step(self, sub, obj, rel, ts, filter_mask=None, copy_mask=None, train=True, log=True, freeze_emb=False, phi_offset=0):
        if train:
            self.train()
            self.optimizer.zero_grad()
            self.copy_optimizer.zero_grad()
        else:
            self.eval()
        if train==False:
            self._deb += 1
        sub_pre, obj_pre, copy_sub_predict, copy_obj_predict = self.forward(sub, obj, rel, ts, copy_mask, freeze_emb=freeze_emb, log=log, phi_offset=phi_offset)
        tru = torch.cat([sub, obj])
        # if train==False:
        #     _csp = copy_sub_predict
        #     _sp = sub_pre
        #     _pcsp = nn.functional.softmax(_csp, dim=1)
        #     _psp = nn.functional.softmax(_sp, dim=1)
        # seperate copy and gen loss to avoid large copy ratio lead to small gradients in gen
        gen_pre = torch.cat([sub_pre, obj_pre])
        copy_pre = torch.cat([copy_sub_predict, copy_obj_predict])
        gen_loss = self.loss_fn(gen_pre, tru)
        copy_loss = self.copy_loss_fn(copy_pre, tru)
        if not self.norm_loss:
            gen_loss = torch.mean(gen_loss)
            copy_loss = torch.mean(copy_loss)
        else:
            norm_fact = torch.cat([self.s_dist[sub],self.o_dist[obj]], dim=0)
            # assert (norm_fact==0).nonzero().shape[0]==0
            norm_fact = norm_fact / torch.sum(norm_fact)
            # import pdb; pdb.set_trace()
            gen_loss = torch.sum(gen_loss * norm_fact)
            copy_loss = torch.sum(copy_loss * norm_fact)
        if train:
            gen_loss.backward()
            copy_loss.backward()
            self.optimizer.step()
            self.copy_optimizer.step()
        # sub_pre = nn.functional.softmax(sub_pre, dim=1)
        # obj_pre = nn.functional.softmax(obj_pre, dim=1)
        # if self.copy > 0:
        #     copy_sub_predict = nn.functional.softmax(copy_sub_predict, dim=1)
        #     copy_obj_predict = nn.functional.softmax(copy_obj_predict, dim=1)
        #     sub_pre = sub_pre * (1 - self.copy) + copy_sub_predict * self.copy
        #     obj_pre = obj_pre * (1 - self.copy) + copy_obj_predict * self.copy
        with torch.no_grad():
            sub_pre += copy_sub_predict
            obj_pre += copy_obj_predict
            pre = torch.cat([sub_pre, obj_pre])
            pre = pre.clone().detach()
            tru = tru.clone().detach()
            pre_thres = pre.gather(1,tru.unsqueeze(1))
            rank_unf = get_rank(pre, pre_thres)
            rank_fil = None
            if filter_mask is not None:
                pre[filter_mask] = float('-inf')
                pre = pre.scatter(1, tru.unsqueeze(1), pre_thres)
                rank_fil = get_rank(pre, pre_thres)
        # if train==False:
        #     if self._deb > 100:
        #         import pdb; pdb.set_trace()
        return gen_loss, rank_unf, rank_fil