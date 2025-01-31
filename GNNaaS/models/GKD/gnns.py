import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse import SparseTensor, matmul
from torch_geometric.nn import GCNConv, SGConv, GATConv, JumpingKnowledge, APPNP, MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.utils import remove_self_loops, add_self_loops, degree
import scipy.sparse
import numpy as np
import math
from copy import copy


class RGCNConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, node_types, edge_types):
        super(RGCNConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        edge_types_dic = {}
        for key in edge_types:
            edge_types_dic[f'{key[0].replace(".","_")}_{key[1].replace(".","_")}_{key[2].replace(".","_")}'] = nn.Linear(in_channels, out_channels, bias=False)
        self.rel_lins = nn.ModuleDict(edge_types_dic)

        self.root_lins = nn.ModuleDict({  ## create linear layer for each node type (distinct veriex i.e author,paper,...)
            key.replace(".","_"): nn.Linear(in_channels, out_channels, bias=True)
            for key in node_types
        })
        self.reset_parameters()

    def reset_parameters(self):
        for lin in self.rel_lins.values():
            lin.reset_parameters()
        for lin in self.root_lins.values():
            lin.reset_parameters()

    def forward(self, x_dict, adj_t_dict,key2int):  ## aggregate updates
        out_dict = {}
        for key, x in x_dict.items():
            out_dict[key] = self.root_lins[key](x)

        for key, adj_t in adj_t_dict.items():
            key_str = f'{key[0]}_{key[1]}_{key[2]}'
            x = x_dict[str(key2int[key[0]])]
            out = self.rel_lins[key_str](adj_t.matmul(x, reduce='mean'))
            out_dict[str(key2int[key[2]])].add_(out)

        return out_dict


class RGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout, num_nodes_dict, x_types, edge_types):
        super(RGCN, self).__init__()

        node_types = list(num_nodes_dict.keys())
        set_diff = set(node_types).difference(set([x_types]))
        param_dic = {}
        for key in set_diff:
            if key not in ['type']:

                param_dic[key.replace(".","_")] = nn.Parameter(torch.Tensor(num_nodes_dict[key], in_channels))

        self.x_dict = None

        self.embs = nn.ParameterDict(param_dic)
        self.convs = nn.ModuleList()
        self.convs.append(
            RGCNConv(in_channels, hidden_channels, node_types, edge_types))  ## Start layer
        for _ in range(num_layers - 2):  ## hidden Layers
            self.convs.append(
                RGCNConv(hidden_channels, hidden_channels, node_types, edge_types))
        self.convs.append(RGCNConv(hidden_channels, out_channels, node_types, edge_types))  ## output layer
        self.dropout = dropout
        self.reset_parameters()

    def reset_parameters(self):
        for emb in self.embs.values():
            torch.nn.init.xavier_uniform_(emb)  ## intialize embeddinga with Xavier uniform dist
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x_dict, adj_t_dict,key2int,layer_no = None):
        x_dict = copy(x_dict)  ## copy x_dict features
        if layer_no is None or layer_no == 0:
            for key, emb in self.embs.items():
                x_dict[key] = emb

        if layer_no is not None:
            # for conv in self.convs[:layer_no]:
            conv = self.convs[layer_no]
            x_dict = conv(x_dict, adj_t_dict, key2int)
            if layer_no != -1:
                for key, x in x_dict.items():
                    x_dict[key] = F.relu(x)  ## relu
                    x_dict[key] = F.dropout(x, p=self.dropout, training=self.training)
            return x_dict
        else:
            for conv in self.convs[:-1]:
                x_dict = conv(x_dict, adj_t_dict,key2int)  ## update features from by convolution layer forward (mean)
                for key, x in x_dict.items():
                    x_dict[key] = F.relu(x)  ## relu
                    x_dict[key] = F.dropout(x, p=self.dropout, training=self.training)  ## dropout some updated features
            return self.convs[-1](x_dict, adj_t_dict,key2int)

    # def backward(self,x_dict, adj_t_dict,key2int,layer_no = None):
    #     import pdb
    #     pdb.set_trace()
    #     return x_dict



class SpecialSpmmFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, indices, values, shape, b):
        assert indices.requires_grad == False
        a = torch.sparse_coo_tensor(indices, values, shape)
        ctx.save_for_backward(a, b)
        ctx.N = shape[0]
        return torch.matmul(a, b)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_values = grad_b = None
        if ctx.needs_input_grad[1]:
            grad_a_dense = grad_output.matmul(b.t())
            edge_idx = a._indices()[0, :] * ctx.N + a._indices()[1, :]
            grad_values = grad_a_dense.view(-1)[edge_idx]
        if ctx.needs_input_grad[3]:
            grad_b = a.t().matmul(grad_output)
        return None, grad_values, None, grad_b


class SpecialSpmm(nn.Module):
    def forward(self, indices, values, shape, b):
        return SpecialSpmmFunction.apply(indices, values, shape, b)



class GCNLayer(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.0, simple=False):
        super(GCNLayer, self).__init__()
        self.simple = simple
        if not simple:
            self.W = nn.Parameter(torch.zeros(in_channels, out_channels))
        self.dropout = dropout
        self.specialspmm = SpecialSpmm()
        self.reset_parameters()

    def reset_parameters(self):
        if not self.simple: nn.init.xavier_uniform_(self.W.data, gain=1.414)

    def forward(self, x, edge_index, ff = True):
        if not self.simple and ff: h = torch.matmul(x, self.W)
        else: h = x
        N = h.size(0)

        # weight_mat: hard and differentiable affinity matrix
        edge_index, _ = remove_self_loops(edge_index)
        edge_index, _ = add_self_loops(edge_index, num_nodes=N)
        src, dst = edge_index

        deg = degree(dst, num_nodes=N)
        deg_src = deg[src].pow_(-0.5)
        deg_src.masked_fill_(deg_src == float('inf'), 0)
        deg_dst = deg[dst].pow_(-0.5)
        deg_dst.masked_fill_(deg_dst == float('inf'), 0)
        edge_weight = deg_src * deg_dst

        h_prime = self.specialspmm(edge_index, edge_weight, torch.Size([N, N]), h)
        return h_prime




class GATLayer(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.0, alpha=0.2):
        super(GATLayer, self).__init__()

        self.W = nn.Parameter(torch.zeros(in_channels, out_channels))
        self.a = nn.Parameter(torch.zeros(1, out_channels * 2))

        self.leakyrelu = nn.LeakyReLU(alpha)
        self.specialspmm = SpecialSpmm()

        self.dropout = dropout
        self.out_channels = out_channels
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

    def forward(self, x, edge_index, edge_weight=None, output_weight=False):
        h = torch.matmul(x, self.W)
        N = h.size(0)
        edge_index, _ = remove_self_loops(edge_index)
        edge_index, _ = add_self_loops(edge_index, num_nodes=N)
        src, dst = edge_index

        # edge_index, _ = remove_self_loops(edge_index)
        # edge_index, _ = add_self_loops(edge_index, num_nodes=N)

        if edge_weight is None:
            edge_h = torch.cat((h[src], h[dst]), dim=-1)  # [E, 2*D]

            edge_e = (edge_h * self.a).sum(dim=-1) # [E]
            edge_e = torch.exp(self.leakyrelu(edge_e))  # [E]
            edge_e = F.dropout(edge_e, p=self.dropout, training=self.training) # [E]
            # e = torch.sparse_coo_tensor(edge_index, edge_e, size=torch.Size([N, N]))
            e_expsum = self.specialspmm(edge_index, edge_e, torch.Size([N, N]), torch.ones(N, 1).to(x.device))
            assert not torch.isnan(e_expsum).any()

            # edge_e_ = F.dropout(edge_e, p=0.8, training=self.training)
            h_prime = self.specialspmm(edge_index, edge_e, torch.Size([N, N]), h)
            h_prime = torch.div(h_prime, e_expsum)  # [N, D] tensor
        else:
            h_prime = self.specialspmm(edge_index, edge_weight, torch.Size([N, N]), h)

        if output_weight:
            edge_expsum = e_expsum[dst].squeeze(1)
            return h_prime, torch.div(edge_e, edge_expsum)
        else:
            return h_prime
