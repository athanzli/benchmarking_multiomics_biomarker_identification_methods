import torch.nn as nn
import torch.nn.functional as F
import math
import torch
import torch.autograd
import numpy as np
import random
import os
from torch.nn.parameter import Parameter
def xavier_init(m):
    if type(m) == nn.Linear:
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
           m.bias.data.fill_(0.0)
           
class Attention(nn.Module):
    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
        # softmax+dropout
        attn = attn / abs(attn.min())
        attn = self.dropout(F.softmax(F.normalize(attn, p=1, dim=-1), dim=-1))
        output = torch.matmul(attn, v)

        return output, attn, v

class VariLengthInputLayer(nn.Module):
    def __init__(self, modal_num, num_class, input_data_dims, d_k, d_v, n_head, dropout, device):
        super(VariLengthInputLayer, self).__init__()
        self.device=device
        self.n_head = n_head
        self.dims = input_data_dims
        self.d_k = d_k
        self.d_v = d_v
        self.w_qs = []
        self.w_ks = []
        self.w_vs = []
        self.modal_num = modal_num
        for i, dim in enumerate(self.dims):
            self.w_q = nn.Linear(dim, n_head * d_k, bias=True)
            self.w_k = nn.Linear(dim, n_head * d_k, bias=True)
            self.w_v = nn.Linear(dim, n_head * d_v, bias=True)
            self.w_qs.append(self.w_q)
            self.w_ks.append(self.w_k)
            self.w_vs.append(self.w_v)
            self.add_module('linear_q_%d_%d' % (dim, i), self.w_q)
            self.add_module('linear_k_%d_%d' % (dim, i), self.w_k)
            self.add_module('linear_v_%d_%d' % (dim, i), self.w_v)

        self.attention = Attention(temperature=d_k ** 0.5, attn_dropout=dropout)
        self.fc = nn.Linear(n_head * d_v, n_head * d_v)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(n_head * d_v, eps=1e-6)
        self.model = nn.Sequential(nn.Linear(self.modal_num*self.n_head*self.d_k, num_class))
        self.model.apply(xavier_init)
        self.to(device)

    def forward(self, input_data, mask=None):

        temp_dim = 0
        bs = input_data.size(0)

        modal_num = len(self.dims)
        q = torch.zeros(bs, modal_num, self.n_head * self.d_k).to(self.device)
        k = torch.zeros(bs, modal_num, self.n_head * self.d_k).to(self.device)
        v = torch.zeros(bs, modal_num, self.n_head * self.d_v).to(self.device)

        for i in range(modal_num):
            w_q = self.w_qs[i]
            w_k = self.w_ks[i]
            w_v = self.w_vs[i]

            data = input_data[:, temp_dim: temp_dim + self.dims[i]]
            temp_dim += self.dims[i]
            q[:, i, :] = w_q(data)
            k[:, i, :] = w_k(data)
            v[:, i, :] = w_v(data)

        q = q.view(bs, modal_num, self.n_head, self.d_k)
        k = k.view(bs, modal_num, self.n_head, self.d_k)
        v = v.view(bs, modal_num, self.n_head, self.d_v)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        q, attn, residual = self.attention(q, k, v)
        q = q.transpose(1, 2).contiguous().view(bs, modal_num, -1)

        residual = residual.transpose(1, 2).contiguous().view(bs, modal_num, -1)
        q = self.dropout(self.fc(q))
        q += residual


        q = self.layer_norm(q)
        q = torch.reshape(q, (-1,self.modal_num*self.n_head*self.d_k))

        output = self.model(q)
        return output

class TransformerEncoder(nn.Module):
    def __init__(self, input_data_dims, hyperpm, num_class, device):
        super(TransformerEncoder, self).__init__()
        self.hyperpm = hyperpm
        self.input_data_dims = input_data_dims
        self.d_q = hyperpm['n_hidden']
        self.d_k = hyperpm['n_hidden']
        self.d_v = hyperpm['n_hidden']
        self.n_head = hyperpm['n_head']
        self.dropout = hyperpm['dropout']
        self.modal_num = hyperpm['nmodal']

        self.n_class = num_class
        self.d_out = self.d_v * self.n_head * self.modal_num

        self.device = device
        self.InputLayer = VariLengthInputLayer(self.modal_num, num_class, self.input_data_dims, self.d_k, self.d_v, self.n_head, self.dropout, device=device)
        self.to(device)

    def forward(self, x):
        bs = x.size(0)

        output = self.InputLayer(x)
        return output
class Classifier_1(nn.Module):
    def __init__(self, in_dim, out_dim, device):
        super().__init__()
        self.clf = nn.Sequential(nn.Linear(in_dim, out_dim))
        self.clf.apply(xavier_init)
        self.to(device)
        self.device = device

    def forward(self, x):
        x = self.clf(x)
        return x





class HGNN_conv(nn.Module):
    def __init__(self, in_ft, out_ft, bias=True, device='cuda'):
        super(HGNN_conv, self).__init__()

        self.weight = Parameter(torch.Tensor(in_ft, out_ft))
        if bias:
            self.bias = Parameter(torch.Tensor(out_ft))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        self.device = device
        self.to(device)

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x: torch.Tensor, G: torch.Tensor):
        G = G.to(self.device)

        x = x.matmul(self.weight)
        if self.bias is not None:
            x = x + self.bias
        x = G.matmul(x)
        return x
class HGNN(nn.Module):
    def __init__(self, in_ch, n_class, n_hid, dropout=0.5, device='cuda'):
        super(HGNN, self).__init__()
        self.dropout = dropout
        self.hgc1 = HGNN_conv(in_ch, n_hid[0], device=device)
        self.hgc2 = HGNN_conv(n_hid[0], n_hid[1], device=device)
        self.device = device

    def forward(self, x, G):
        x = self.hgc1(x, G)
        x = F.leaky_relu(x,0.25)
        x = F.dropout(x, self.dropout)
        x = self.hgc2(x, G)
        x = F.leaky_relu(x, 0.25)
        return x

def init_model_dict(input_data_dims, hyperpm, num_view, num_class, dim_list, dim_he_list, dim_hc, gcn_dopout=0.5, device='cuda'):
    model_dict = {}
    for i in range(num_view):
        model_dict["E{:}".format(i+1)] = HGNN(dim_list[i], num_class, dim_he_list, gcn_dopout, device)
        model_dict["C{:}".format(i+1)] = Classifier_1(dim_he_list[-1], num_class, device)
    if num_view >= 2:
        model_dict["C"] = TransformerEncoder(input_data_dims, hyperpm, num_class, device)
    return model_dict


def init_optim(num_view, model_dict, lr_e, lr_c):
    optim_dict = {}
    for i in range(num_view):
        optim_dict["C{:}".format(i+1)] = torch.optim.Adam(
                list(model_dict["E{:}".format(i+1)].parameters())+list(model_dict["C{:}".format(i+1)].parameters()), 
                lr=lr_e)
    if num_view >= 2:
        optim_dict["C"] = torch.optim.Adam(model_dict["C"].parameters(), lr=lr_c)
    return optim_dict