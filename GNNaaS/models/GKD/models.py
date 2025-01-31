import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse import SparseTensor, matmul
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.utils import num_nodes, to_dense_adj



from gnns import *
import numpy as np
from kernels import Kernel
# from GNNaaS.models.graph_saint_Shadow_KGTOSA import RGCN
# from GNNaaS.models.rgcn_KGTOSA import RGCN
from gnns import RGCN
from copy import copy
class GeoDist(nn.Module):
    '''
    in_channels: number of features 
    hidden_channels: hidden size
    '''
    def __init__(self, in_channels, hidden_channels, out_channels, args, num_node, num_layers=2,hetero_params=[],
                 type='rgcn',use_bn=False):
        super(GeoDist, self).__init__()
        self.args = args
        self.dropout = args.dropout
        self.num_node = num_node
        self.k = Kernel(hidden_channels, out_channels, args, num_node)
        self.type = type
        self.teacher_gnn = nn.ModuleList([nn.Linear(in_channels, hidden_channels, bias=False)])
        self.student_gnn = nn.ModuleList([nn.Linear(in_channels, hidden_channels, bias=False)])
        self.bns = nn.BatchNorm1d(hidden_channels, eps=1e-10, affine=False, track_running_stats=False)
        self.bns2d = nn.BatchNorm2d(hidden_channels, affine=False, track_running_stats=False)

        if self.type == 'gcn':
            self.teacher_gnn.append(GCNLayer(hidden_channels, hidden_channels, dropout=self.dropout, simple=True))
            self.student_gnn.append(GCNLayer(hidden_channels, hidden_channels, dropout=self.dropout, simple=True))
        
            for _ in range(num_layers - 2):
                self.teacher_gnn.append(
                        GCNLayer(hidden_channels, hidden_channels))
                self.student_gnn.append(
                    GCNLayer(hidden_channels, hidden_channels))

            self.teacher_gnn.append(GCNLayer(hidden_channels, out_channels))
            self.student_gnn.append(GCNLayer(hidden_channels, out_channels))
            
        elif self.type == 'gat':
            self.teacher_gnn.append(GATLayer(hidden_channels, hidden_channels, dropout=self.dropout, simple=True))
            self.student_gnn.append(GATLayer(hidden_channels, hidden_channels, dropout=self.dropout, simple=True))
            
            for _ in range(num_layers - 2):
                self.teacher_gnn.append(
                        GATLayer(hidden_channels, hidden_channels, dropout=self.dropout))
                self.student_gnn.append(
                    GATLayer(hidden_channels, hidden_channels, dropout=self.dropout))
            
            self.teacher_gnn.append(GATLayer(hidden_channels, out_channels, dropout=self.dropout))
            self.student_gnn.append(GATLayer(hidden_channels, out_channels, dropout=self.dropout))

        elif self.type == 'rgcn':
            num_node_types,x_types,num_edge_types, = hetero_params
            # New RGCN initialization for teacher and student models
            self.teacher_gnn = RGCN(in_channels, hidden_channels, out_channels, num_layers,
                                    self.dropout, num_node_types, x_types, num_edge_types)
            self.student_gnn = RGCN(in_channels, hidden_channels, out_channels, num_layers,
                                    self.dropout, num_node_types, x_types, num_edge_types)


        self.activation = F.relu
        self.use_bn = use_bn

    def reset_parameters(self):
        for conv in self.teacher_gnn.convs:
            conv.reset_parameters()
        for conv in self.student_gnn.convs:
            conv.reset_parameters()

    def forward(self, data_full, data=None, mode='pretrain', dist_mode='label', t=1.0):
        if mode == 'pretrain':
            return self.forward_teacher(data_full)
        elif mode == 'train':  
            return self.forward_student(data_full, data, dist_mode, t)
        else:
            NotImplementedError
    
    def forward_teacher(self, data_full):
        if self.type == 'rgcn':
            # x_dict, (edge_index, edge_type, node_type, local_node_idx) = data_full
            x_dict,data,key2int = data_full

            # x_dict,train_loader = data_full
            # for data
            # for G-SAINT sampler
            # x = self.teacher_gnn(x_dict, data.edge_index, data.edge_attr, data.node_type, data.local_node_idx)
            x = self.teacher_gnn(x_dict, data,key2int)

        else:
            x, edge_index = data_full.graph['node_feat'], data_full.graph['edge_index']
            x = self.teacher_gnn[0](x)
            for i in range(1, len(self.teacher_gnn) - 1):
                x = self.teacher_gnn[i](x, edge_index) # [n, h]
                if self.use_bn: x = self.bns(x)
                x = self.activation(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.teacher_gnn[-1](x, edge_index)
        return x
    
    def forward_student(self, data_full, data, dist_mode, t = 1):
        if self.type == 'rgcn':
            # print('here')
            xt = xs = data_full[0]# copy(data_full[0])#xs = data_full[0] # x_dict
            # for key,emb in self.teacher_gnn.embs.items():
            #     xt[key] = emb
            # for key,emb in self.student_gnn.embs.items():
            #     xs[key] = emb

            edge_index  = data[0] # data.graph['edge_index']
            train_idx = data[1]
            share_node_idx = data[2]
            edge_index_full = data_full[1]#.edge_index for G-SAINT
            key2int = data_full[-1]
            subject_node = data_full[-2]

        else:
            x_full, edge_index_full = data_full.graph['node_feat'], data_full.graph['edge_index']
            xt = xs = x_full # edge missing, node is seen for both teacher and student
            edge_index = data.graph['edge_index']

            if self.args.use_batch:
                idx = torch.randperm(len(data.share_node_idx))[:self.args.batch_size]
                share_node_idx = data.share_node_idx[idx]
            else: share_node_idx = data.share_node_idx
            train_idx = data.train_idx
        
        if dist_mode == 'no': 
            return self.no_kd(xs, xt, edge_index, edge_index_full, train_idx, share_node_idx)
        elif dist_mode == 'gkd':
            return self.gkd(xs, xt, edge_index, edge_index_full, train_idx, share_node_idx,key2int=key2int,subject_node=subject_node)
        elif dist_mode == 'pgkd':
            return self.pgkd(xs, xt, edge_index, edge_index_full, train_idx, share_node_idx)
        else:
            NotImplementedError
            
    def inference(self, data, mode='pretrain'):
        # x, edge_index = data.graph['node_feat'], data.graph['edge_index']
        x_dict, edge_index_dict,key2int = data#data.graph['node_feat'], data.graph['edge_index']
        if mode == 'pretrain':
            # x = self.teacher_gnn.convs[0](x_dict, edge_index_dict,edge_type,node_type)
            # for i in range(1, len(self.teacher_gnn.convs) - 1):
            #     x = self.teacher_gnn[i](x, edge_index_dict,edge_type,node_type)
            #     if self.use_bn:
            #         if 'share_node_idx' in data.__dict__.keys():
            #             x[data.share_node_idx] = self.bns(x[data.share_node_idx])
            #         else: x = self.bns(x)
            #     x = self.activation(x)
            # x = self.teacher_gnn[-1](x, edge_index_dict,edge_type,node_type)
            # x = self.teacher_gnn.inference(x_dict,edge_index_dict,key2int) # FOR G-SAINT
            x = self.teacher_gnn(x_dict,edge_index_dict,key2int)
            return x

        elif mode == 'train':
            edge_index_dict , share_node_idx = edge_index_dict#data.share_node_idx
            x = self.student_gnn(x_dict,edge_index_dict,key2int)
            # x = self.student_gnn.convs[0](x)
            # for i in range(1, len(self.student_gnn) - 1):
            #     x = self.student_gnn[i](x, edge_index_dict)
            #     if self.use_bn: x[share_node_idx] = self.bns(x[share_node_idx])
            #     x = self.activation(x)
            # x = self.student_gnn[-1](x, edge_index_dict)
            return x
    #
    # def inference(self, x_dict, edge_index_dict, key2int):
    #     device = list(x_dict.values())[0].device
    #     x_dict = copy(x_dict)
    #     for key, emb in self.emb_dict.items():
    #         x_dict[int(key)] = emb
    #
    #     adj_t_dict = {}
    #     for key, (row, col) in edge_index_dict.items():
    #         adj_t_dict[key] = SparseTensor(row=col, col=row).to(device)
    #     for i, conv in enumerate(self.convs):
    #         out_dict = {}
    #
    #         for j, x in x_dict.items():
    #             out_dict[j] = conv.root_lins[j](x)
    #
    #         for keys, adj_t in adj_t_dict.items():
    #             src_key, target_key = keys[0], keys[-1]
    #             out = out_dict[key2int[target_key]]
    #             tmp = adj_t.matmul(x_dict[key2int[src_key]], reduce='mean')
    #             tmp = conv.rel_lins[key2int[keys]](tmp).resize_([out.size()[0], out.size()[1]])
    #             out.add_(tmp)
    #
    #         if i != self.num_layers - 1:
    #             for j in range(self.num_node_types):
    #                 F.relu_(out_dict[j])
    #
    #         x_dict = out_dict
    #
    #     return x_dict

        
    def no_kd(self, xs, xt, edge_index, edge_index_full, train_idx, share_node_idx):
        xs = self.student_gnn[0](xs)
        for i in range(1, len(self.student_gnn) - 1):
            xs = self.student_gnn[i](xs, edge_index)
            xs[share_node_idx] = self.bns(xs[share_node_idx])
            xs = self.activation(xs)
        y_logit_s = self.student_gnn[-1](xs, edge_index)
        return y_logit_s
    
    def gkd(self, xs, xt, edge_index, edge_index_full, train_idx, share_node_idx,key2int,subject_node):
        # if self.args.delta != 1:
        #     A = to_dense_adj(edge_index, max_num_nodes=self.num_node).squeeze().fill_diagonal_(1.)
        #     A = A[share_node_idx, :][:, share_node_idx]
        # else: A = None
        A = None
        loss_list = []
        # xt = self.teacher_gnn.convs[0](xt,edge_index_full,key2int)
        # xs = self.student_gnn.convs[0](xs,edge_index,key2int)
        xt = self.teacher_gnn(xt,edge_index_full,key2int,layer_no=0)
        xs = self.student_gnn(xs,edge_index,key2int,layer_no=0) # edge_index

        # Storing non-target intermediate node embeddings (for second layer)
        non_target_teacher = {}
        for k,emb in xt.items():
            if k != str(key2int[subject_node]):
                non_target_teacher[k] = emb

        non_target_student = {}
        for k, emb in xt.items():
            if k != str(key2int[subject_node]):
                non_target_student[k] = emb

        xt,xs = xt[str(key2int[subject_node])],xs[str(key2int[subject_node])]
        # xt, xs = self.bns(xt), self.bns(xs)
        if self.args.kernel != 'random':
            "rewriting batched version to avoid OOMs"
            mt = self.k(xt[share_node_idx], xt[share_node_idx]).detach()
            ms = self.k(xs[share_node_idx], xs[share_node_idx])

            # mt = torch.zeros((share_node_idx.shape[0],share_node_idx.shape[0]))
            # ms = torch.zeros((share_node_idx.shape[0],share_node_idx.shape[0]))
            # for i in (0,len(share_node_idx),self.args.batch_size):
            #     batch_mt = self.k(xt[share_node_idx][i:i+self.args.batch_size],
            #                       xt[share_node_idx][i:i+self.args.batch_size]).detach()
            #     mt[i:i+self.args.batch_size,:] = batch_mt
            #
            #     batch_ms = self.k(xs[share_node_idx][i:i + self.args.batch_size],
            #                       xs[share_node_idx][i:i + self.args.batch_size])
            #     ms[i:i + self.args.batch_size, :] = batch_ms


        else:
            mt, ms = self.k.random(xt[share_node_idx], xs[share_node_idx])
        loss_list.append(self.k.dist_loss(mt, ms, A))
        xt, xs = self.activation(xt), self.activation(xs)
        # for i in range(1, len(self.teacher_gnn.convs) - 1):
        #     xt = self.teacher_gnn.convs[i](xt, edge_index_full,key2int)
        #     xs = self.student_gnn.convs[i](xs, edge_index,key2int)
        #     xt,xs = xt[str(key2int[subject_node])] , xs[str(key2int[subject_node])]
        #     xt, xs = self.bns(xt), self.bns(xs)
        #     if self.args.kernel != 'random':
        #         mt = self.k(xt[share_node_idx], xt[share_node_idx]).detach()
        #         ms = self.k(xs[share_node_idx], xs[share_node_idx])
        #     else:
        #         mt, ms = self.k.random(xt[share_node_idx], xs[share_node_idx])
        #     loss_list.append(self.k.dist_loss(mt, ms, A))
        #     xt, xs = self.activation(xt), self.activation(xs)

        # Re-adding Non-target nodes
        xt,xs = {str(key2int[subject_node]):xt} , {str(key2int[subject_node]):xs}
        for k,emb in non_target_teacher.items():
            xt[k] = emb
        for k, emb in non_target_student.items():
            xs[k] = emb
        del non_target_teacher, non_target_student,emb
        # y_logit_t = self.teacher_gnn.convs[-1](xt, edge_index_full,key2int)
        # y_logit_s = self.student_gnn.convs[-1](xs, edge_index,key2int)
        y_logit_t = self.teacher_gnn(xt, edge_index_full,key2int,layer_no=-1)
        y_logit_s = self.student_gnn(xs, edge_index,key2int,layer_no=-1) # edge_index
        y_logit_t,y_logit_s = y_logit_t[str(key2int[subject_node])] , y_logit_s[str(key2int[subject_node])]
        if self.args.include_last:
            if self.args.kernel != 'random':
                mt = self.k(y_logit_t[share_node_idx], y_logit_t[share_node_idx]).detach()
                ms = self.k(y_logit_s[share_node_idx], y_logit_s[share_node_idx])
            else:
                mt, ms = self.k.random(y_logit_t[share_node_idx], y_logit_s[share_node_idx])
            loss_list.append(self.k.dist_loss(mt, ms, A))
        gkd_dist_loss = sum(loss_list)/len(loss_list)
        if self.args.use_kd:
            dist_loss = nn.KLDivLoss(reduction='batchmean')(F.log_softmax(y_logit_s[train_idx]/self.args.tau, dim=1), F.softmax(y_logit_t[train_idx].detach()/self.args.tau, dim=1))
            return y_logit_s, gkd_dist_loss, dist_loss
        else:
            return y_logit_s, gkd_dist_loss

    def pgkd(self, xs, xt, edge_index, edge_index_full, train_idx, share_node_idx):
        if self.args.delta != 1:
            A = to_dense_adj(edge_index, max_num_nodes=self.num_node).squeeze().fill_diagonal_(1.)
            A = A[share_node_idx, :][:, share_node_idx]
        else: A = None
        
        xt = self.teacher_gnn[0](xt)
        xs = self.student_gnn[0](xs)    
        xt1, xs1 = xt[share_node_idx].clone(), xs[share_node_idx].clone()
            
        for i in range(1, len(self.teacher_gnn) - 1):
            xt = self.teacher_gnn[i](xt, edge_index_full)
            xs = self.student_gnn[i](xs, edge_index)
            xt, xs = self.bns(xt), self.bns(xs)
            xt, xs = self.activation(xt), self.activation(xs)

        y_logit_t = self.teacher_gnn[-1](xt, edge_index_full, ff=False)
        y_logit_s = self.student_gnn[-1](xs, edge_index, ff=False)
        
        y_logit_t0 = y_logit_t[share_node_idx]
        y_logit_s0 = y_logit_s[share_node_idx]
        
        mt, ms = self.k.parametric(y_logit_t0.detach(), y_logit_s0, detach=True)
        gkd_dist_loss = self.k.dist_loss(mt, ms, A)
        
        mt2, ms2 = self.k.parametric(y_logit_t0, y_logit_s0)
        rec_loss =  self.k.rec_loss(y_logit_t0.detach(), xt1.detach(), mt2) 
        rec_loss += self.k.rec_loss(y_logit_s0.detach(), xs1.detach(), ms2) 
        
        y_logit_t_, y_logit_s_ = torch.matmul(y_logit_t, self.teacher_gnn[-1].W), torch.matmul(y_logit_s, self.student_gnn[-1].W)
        if self.args.use_kd:
            dist_loss = nn.KLDivLoss(reduction='batchmean')(F.log_softmax(y_logit_s[train_idx]/self.args.tau, dim=1), F.softmax(y_logit_t[train_idx].detach()/self.args.tau, dim=1))
            return y_logit_s_, gkd_dist_loss, rec_loss, dist_loss
        else:
            return y_logit_s_, gkd_dist_loss, rec_loss