""""
Implements the link prediction task on the FB15k237 datasets according to the
`"Modeling Relational Data with Graph Convolutional Networks"
<https://arxiv.org/abs/1703.06103>`_ paper.

Caution: This script is executed in a full-batch fashion, and therefore needs
to run on CPU (following the experimental setup in the official paper).
"""
import sys
GMLaaS_models_path=sys.path[0].split("KGNET")[0]+"/KGNET/GNNaaS/models/rgcn"
sys.path.insert(0,GMLaaS_models_path)
from Constants import *
#import argparse
import torch
import torch.nn.functional as F
from torch.nn import Parameter
from tqdm.auto import tqdm
from rel_link_pred_dataset import RelLinkPredDataset
#from torch_geometric.datasets import RelLinkPredDataset
from torch_geometric.nn import GAE, RGCNConv
from resource import *
import datetime
import os
import json
import pandas as pd
import os.path as osp
from torch_geometric.nn.conv import MessagePassing
from typing import Optional, Tuple, Union
from torch.nn import Parameter as Param
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.typing import (
    Adj,OptTensor,SparseTensor,pyg_lib,torch_sparse,
)
from torch import Tensor
from torch_geometric.utils import index_sort, one_hot, scatter, spmm
import torch_geometric.typing
from torch_geometric.utils.sparse import index2ptr

#from utils import calc_mrr
#path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'RLPD')
#dataset = RelLinkPredDataset(path, 'FB15k-237')


def gen_model_name(dataset_name='',GNN_Method=''):
    return dataset_name
def create_dir(list_paths):
    for path in list_paths:
        if not os.path.exists(path):
            os.mkdir(path)

@torch.jit._overload
def masked_edge_index(edge_index, edge_mask):
    # type: (Tensor, Tensor) -> Tensor
    pass


@torch.jit._overload
def masked_edge_index(edge_index, edge_mask):
    # type: (SparseTensor, Tensor) -> SparseTensor
    pass


def masked_edge_index(edge_index, edge_mask):
    if isinstance(edge_index, Tensor):
        return edge_index[:, edge_mask]
    return torch_sparse.masked_select_nnz(edge_index, edge_mask, layout='coo')

class RGCNConv(MessagePassing):

    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        num_relations: int,
        num_bases: Optional[int] = None,
        num_blocks: Optional[int] = None,
        aggr: str = 'mean',
        root_weight: bool = True,
        is_sorted: bool = False,
        bias: bool = True,
        **kwargs,
    ):
        kwargs.setdefault('aggr', aggr)
        super().__init__(node_dim=0, **kwargs)

        if num_bases is not None and num_blocks is not None:
            raise ValueError('Can not apply both basis-decomposition and '
                             'block-diagonal-decomposition at the same time.')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_relations = num_relations
        self.num_bases = num_bases
        self.num_blocks = num_blocks
        self.is_sorted = is_sorted

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)
        self.in_channels_l = in_channels[0]

        if num_bases is not None:
            self.weight = Parameter(
                torch.Tensor(num_bases, in_channels[0], out_channels))
            self.comp = Parameter(torch.Tensor(num_relations, num_bases))

        elif num_blocks is not None:
            assert (in_channels[0] % num_blocks == 0
                    and out_channels % num_blocks == 0)
            self.weight = Parameter(
                torch.Tensor(num_relations, num_blocks,
                             in_channels[0] // num_blocks,
                             out_channels // num_blocks))
            self.register_parameter('comp', None)

        else:
            self.weight = Parameter(
                torch.Tensor(num_relations, in_channels[0], out_channels))
            self.register_parameter('comp', None)

        if root_weight:
            self.root = Param(torch.Tensor(in_channels[1], out_channels))
        else:
            self.register_parameter('root', None)

        if bias:
            self.bias = Param(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        glorot(self.weight)
        glorot(self.comp)
        glorot(self.root)
        zeros(self.bias)

    def forward(self, x: Union[OptTensor, Tuple[OptTensor, Tensor]],
                edge_index: Adj, edge_type: OptTensor = None):

        # Convert input features to a pair of node features or node indices.
        x_l: OptTensor = None
        if isinstance(x, tuple):
            x_l = x[0]
        else:
            x_l = x
        if x_l is None:
            x_l = torch.arange(self.in_channels_l, device=self.weight.device)

        x_r: Tensor = x_l
        if isinstance(x, tuple):
            x_r = x[1]

        size = (x_l.size(0), x_r.size(0))

        if isinstance(edge_index, SparseTensor):
            edge_type = edge_index.storage.value()
        assert edge_type is not None

        # propagate_type: (x: Tensor, edge_type_ptr: OptTensor)
        out = torch.zeros(x_r.size(0), self.out_channels, device=x_r.device)

        weight = self.weight
        if self.num_bases is not None:  # Basis-decomposition =================
            weight = (self.comp @ weight.view(self.num_bases, -1)).view(
                self.num_relations, self.in_channels_l, self.out_channels)

        if self.num_blocks is not None:  # Block-diagonal-decomposition =====

            if not torch.is_floating_point(
                    x_r) and self.num_blocks is not None:
                raise ValueError('Block-diagonal decomposition not supported '
                                 'for non-continuous input features.')

            for i in range(self.num_relations):
                tmp = masked_edge_index(edge_index, edge_type == i)
                h = self.propagate(tmp, x=x_l, edge_type_ptr=None, size=size)
                h = h.view(-1, weight.size(1), weight.size(2))
                h = torch.einsum('abc,bcd->abd', h, weight[i])
                out = out + h.contiguous().view(-1, self.out_channels)

        else:  # No regularization/Basis-decomposition ========================
            if (torch_geometric.typing.WITH_PYG_LIB and self.num_bases is None
                    and x_l.is_floating_point()
                    and isinstance(edge_index, Tensor)):
                if not self.is_sorted:
                    if (edge_type[1:] < edge_type[:-1]).any():
                        edge_type, perm = index_sort(
                            edge_type, max_value=self.num_relations)
                        edge_index = edge_index[:, perm]
                edge_type_ptr = index2ptr(edge_type, self.num_relations)
                out = self.propagate(edge_index, x=x_l,
                                     edge_type_ptr=edge_type_ptr, size=size)
            else:
                for i in range(self.num_relations):
                    tmp = masked_edge_index(edge_index, edge_type == i)

                    if not torch.is_floating_point(x_r):
                        out = out + self.propagate(
                            tmp,
                            x=weight[i, x_l],
                            edge_type_ptr=None,
                            size=size,
                        )
                    else:
                        h = self.propagate(tmp, x=x_l, edge_type_ptr=None,
                                           size=size)
                        out = out + (h @ weight[i])

        root = self.root
        if root is not None:
            if not torch.is_floating_point(x_r):
                out = out + root[x_r]
            else:
                out = out + x_r @ root

        if self.bias is not None:
            out = out + self.bias

        return out

    def message(self, x_j: Tensor, edge_type_ptr: OptTensor) -> Tensor:
        if torch_geometric.typing.WITH_PYG_LIB and edge_type_ptr is not None:
            # TODO Re-weight according to edge type degree for `aggr=mean`.
            return pyg_lib.ops.segment_matmul(x_j, edge_type_ptr, self.weight)

        return x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        adj_t = adj_t.set_value(None)
        return spmm(adj_t, x, reduce=self.aggr)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, num_relations={self.num_relations})')


class RGCNEncoder(torch.nn.Module):
    def __init__(self, num_nodes, hidden_channels, num_relations):
        super().__init__()
        self.node_emb = Parameter(torch.Tensor(num_nodes, hidden_channels))
        self.conv1 = RGCNConv(hidden_channels, hidden_channels, num_relations,
                              num_blocks=5,
                              )
        self.conv2 = RGCNConv(hidden_channels, hidden_channels, num_relations,
                              num_blocks=5,
                              )
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.node_emb)
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, edge_index, edge_type):
        x = self.node_emb
        x = self.conv1(x, edge_index, edge_type).relu_()
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv2(x, edge_index, edge_type)
        return x


class DistMultDecoder(torch.nn.Module):
    def __init__(self, num_relations, hidden_channels):
        super().__init__()
        self.rel_emb = Parameter(torch.Tensor(num_relations, hidden_channels))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.rel_emb)

    def forward(self, z, edge_index, edge_type):
        z_src, z_dst = z[edge_index[0]], z[edge_index[1]]
        rel = self.rel_emb[edge_type]
        return torch.sum(z_src * rel * z_dst, dim=1)



def negative_sampling(edge_index, num_nodes):
    # Sample edges by corrupting either the subject or the object of each edge.
    mask_1 = torch.rand(edge_index.size(1)) < 0.5
    mask_2 = ~mask_1

    neg_edge_index = edge_index.clone()
    neg_edge_index[0, mask_1] = torch.randint(num_nodes, (mask_1.sum(), ))
    neg_edge_index[1, mask_2] = torch.randint(num_nodes, (mask_2.sum(), ))
    return neg_edge_index


def train():
    model.train()
    optimizer.zero_grad()

    z = model.encode(data.edge_index, data.edge_type)

    pos_out = model.decode(z, data.train_edge_index, data.train_edge_type)

    neg_edge_index = negative_sampling(data.train_edge_index, data.num_nodes)
    neg_out = model.decode(z, neg_edge_index, data.train_edge_type)

    out = torch.cat([pos_out, neg_out])
    gt = torch.cat([torch.ones_like(pos_out), torch.zeros_like(neg_out)])
    cross_entropy_loss = F.binary_cross_entropy_with_logits(out, gt)
    reg_loss = z.pow(2).mean() + model.decoder.rel_emb.pow(2).mean()
    loss = cross_entropy_loss + 1e-2 * reg_loss

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
    optimizer.step()

    return float(loss)


@torch.no_grad()
def test():
    model.eval()
    z = model.encode(data.edge_index, data.edge_type)

    valid_mrr,valid_hits_10 = compute_mrr(z, data.valid_edge_index, data.valid_edge_type)
    test_mrr,test_hits_10 = compute_mrr(z, data.test_edge_index, data.test_edge_type)

    return valid_mrr,valid_hits_10 , test_mrr,test_hits_10


@torch.no_grad()
def compute_rank(ranks):
    # fair ranking prediction as the average
    # of optimistic and pessimistic ranking
    true = ranks[0]
    optimistic = (ranks > true).sum() + 1
    pessimistic = (ranks >= true).sum()
    return (optimistic + pessimistic).float() * 0.5


@torch.no_grad()
def compute_mrr(z, edge_index, edge_type):
    ranks = []
    for i in tqdm(range(edge_type.numel())):
        (src, dst), rel = edge_index[:, i], edge_type[i]

        # Try all nodes as tails, but delete true triplets:
        tail_mask = torch.ones(data.num_nodes, dtype=torch.bool)
        for (heads, tails), types in [
            (data.train_edge_index, data.train_edge_type),
            (data.valid_edge_index, data.valid_edge_type),
            (data.test_edge_index, data.test_edge_type),
        ]:
            tail_mask[tails[(heads == src) & (types == rel)]] = False

        tail = torch.arange(data.num_nodes)[tail_mask]
        tail = torch.cat([torch.tensor([dst]), tail])
        head = torch.full_like(tail, fill_value=src)
        eval_edge_index = torch.stack([head, tail], dim=0)
        eval_edge_type = torch.full_like(tail, fill_value=rel)

        out = model.decode(z, eval_edge_index, eval_edge_type)
        rank = compute_rank(out)
        ranks.append(rank)

        # Try all nodes as heads, but delete true triplets:
        head_mask = torch.ones(data.num_nodes, dtype=torch.bool)
        for (heads, tails), types in [
            (data.train_edge_index, data.train_edge_type),
            (data.valid_edge_index, data.valid_edge_type),
            (data.test_edge_index, data.test_edge_type),
        ]:
            head_mask[heads[(tails == dst) & (types == rel)]] = False

        head = torch.arange(data.num_nodes)[head_mask]
        head = torch.cat([torch.tensor([src]), head])
        tail = torch.full_like(head, fill_value=dst)
        eval_edge_index = torch.stack([head, tail], dim=0)
        eval_edge_type = torch.full_like(head, fill_value=rel)

        out = model.decode(z, eval_edge_index, eval_edge_type)
        rank = compute_rank(out)
        ranks.append(rank)
    ranks = torch.tensor(ranks, dtype=torch.float)
    hits=[10]
    for hit in hits:
        avg_count = torch.mean((ranks <= hit).float())
        hits_10 = avg_count.item()
        print("Hits (filtered) @ {}: {:.6f}".format(hit, hits_10))


    return (1. / torch.tensor(ranks, dtype=torch.float)).mean(),hits_10



def rgcn_lp(dataset_name,
            root_path=KGNET_Config.datasets_output_path,
            epochs=3,val_interval=2,
            hidden_channels=10,batch_size=-1,runs=1,
            emb_size=128,walk_length = 2, num_steps=2,
            loadTrainedModel=0,
            target_rel = '',
            list_src_nodes = [],
            K = 1,modelID = ''
            ):
    # parser = argparse.ArgumentParser()
    # parser.add_argument('root_path',type=str,default=KGNET_Config.datasets_output_path)

    # path = '/home/afandi/GitRepos/KGNET/Datasets/'
    # dataset_name = 'FB15K23'
    init_ru_maxrss = getrusage(RUSAGE_SELF).ru_maxrss
    dict_results = {}
    print('loading dataset..')
    start_data_t = datetime.datetime.now()
    dataset = RelLinkPredDataset(root_path, dataset_name)
    global data,model,optimizer
    data = dataset[0]
    load_data_t = str((datetime.datetime.now()-start_data_t).total_seconds())
    dict_results['Sampling_Time'] = load_data_t
    print(f'dataset loaded at {load_data_t}')

    print('Initializing model...')
    model = GAE(
        RGCNEncoder(data.num_nodes, hidden_channels=hidden_channels,
                    num_relations=dataset.num_relations),
        DistMultDecoder(dataset.num_relations // 2, hidden_channels=hidden_channels),
    )
    print('Model Initialized!')


    if loadTrainedModel == 1:

        ###### LOAD ENTITIES AND REL DICT ##########
        entities_df = pd.read_csv(osp.join(dataset.raw_dir, 'entities.dict'), header=None, sep="\t")
        entities_dict = dict(zip(entities_df[1], entities_df[0]))
        rev_entities_dict = {v:k for k,v in entities_dict.items()}

        relations_df = pd.read_csv(osp.join(dataset.raw_dir, 'relations.dict'), header=None, sep="\t")
        relations_dict = dict(zip(relations_df[1], relations_df[0]))
        if target_rel not in relations_dict:
            return {'error':['Unseen relation']}
        target_rel_ID = relations_dict[target_rel]

        ####### LOAD MODEL STATE ##############
        trained_model_path = os.path.join(KGNET_Config.trained_model_path,modelID)
        model.load_state_dict(torch.load(trained_model_path)); print(f'LOADED PRE-TRAINED MODEL {modelID}')

        with torch.no_grad():
            edge_index = data.edge_index
            edge_type = data.edge_type
            y_pred = {}
            z = model.encode(data.edge_index, data.edge_type)
            # out = model.decode(z,data.edge_index,data.edge_type)
            # print(out)

            # for i in tqdm(range(edge_type.numel())):
            for _src in list_src_nodes:
                if _src not in entities_dict:
                    y_pred[_src] = ['None']
                    continue
                src = entities_dict[_src]
                (_, dst), rel = edge_index[:, target_rel_ID], edge_type[target_rel_ID]

                # Try all nodes as tails, but delete true triplets:
                # tail_mask = torch.ones(data.num_nodes, dtype=torch.bool)
                # for (heads, tails), types in [
                #     (data.train_edge_index, data.train_edge_type),
                #     (data.valid_edge_index, data.valid_edge_type),
                #     (data.test_edge_index, data.test_edge_type),
                # ]:
                #     tail_mask[tails[(heads == src) & (types == rel)]] = False
                #
                # tail = torch.arange(data.num_nodes)[tail_mask]
                tail = torch.arange(data.num_nodes)
                tail = torch.cat([torch.tensor([dst]), tail])
                head = torch.full_like(tail, fill_value=src)
                eval_edge_index = torch.stack([head, tail], dim=0)
                eval_edge_type = torch.full_like(tail, fill_value=rel)

                out = model.decode(z, eval_edge_index, eval_edge_type)
                y_pred[_src] = [rev_entities_dict[i] for i in out.topk(K).indices.tolist() if i in rev_entities_dict]
                # print(out)

        return y_pred



    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    model_loaded_ru_maxrss = getrusage(RUSAGE_SELF).ru_maxrss
    start_train_t = datetime.datetime.now()
    best_test_mrr = 0
    best_valid_mrr = 0
    best_test_hits = 0
    best_valid_hits = 0
    for epoch in tqdm (range(1, epochs),desc='Training Epochs'):
        # print('Starting training .. ')
        loss = train()
        print(f'Epoch: {epoch:05d}, Loss: {loss:.4f}')
        if (epoch % val_interval) == 0:
            valid_mrr,valid_hits_10, test_mrr,test_hits_10 = test()
            print(f'Val MRR: {valid_mrr:.4f}, Val Hits@10: {valid_hits_10:.4f}, Test MRR: {test_mrr:.4f}, Test Hits@10: {test_hits_10:.4f}')
            if test_mrr > best_test_mrr:
                best_test_mrr = test_mrr
                best_test_hits = test_hits_10

            if valid_mrr > best_valid_mrr:
                best_valid_mrr = valid_mrr
                best_valid_hits = valid_hits_10
            print(getrusage(RUSAGE_SELF))
    model_trained_ru_maxrss = getrusage(RUSAGE_SELF).ru_maxrss
    dict_results['Results'] = {'Best_MRR' : best_test_mrr.item(),'Best_Hits@10' : best_test_hits}
    print(getrusage(RUSAGE_SELF))
    end_train_t = datetime.datetime.now()
    total_train_t = str((end_train_t - start_train_t).total_seconds())
    total_time = str((start_data_t - end_train_t).total_seconds())
    dict_results['dataset_name'] = dataset_name
    model_name = gen_model_name(dataset_name)
    dict_results['model_name'] = model_name
    dict_results['Train_Time'] = total_train_t
    dict_results['Total_Time'] = total_time
    dict_results["Model_Parameters_Count"] = sum(p.numel() for p in model.parameters())
    dict_results["Model_Trainable_Paramters_Count"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    dict_results["init_ru_maxrss"] = init_ru_maxrss
    dict_results["model_ru_maxrss"] = model_loaded_ru_maxrss
    dict_results["model_trained_ru_maxrss"] = model_trained_ru_maxrss
    dict_results["Highest_Test_MRR"] = best_test_mrr.item()
    dict_results["Highest_Valid_MRR"] = best_valid_mrr.item()
    dict_results["Highest_Test_Hits@10"] = str(best_test_hits)
    dict_results["Highest_Valid_Hits@10"] = str(best_valid_hits)
    dict_results["Final_Test_MRR"] = test_mrr.item()
    dict_results["Final_Valid_MRR"] = valid_mrr.item()
    dict_results["Final_Test_Hits@10"] = str(test_hits_10)
    dict_results["Final_Valid_Hits@10"] = str(valid_hits_10)
    gnn_hyper_params_dict = { "hidden_channels": hidden_channels,
                               "epochs": epochs,
                              #"runs": runs,
                              #"batch_size": batch_size,
                            #"walk_length": walk_length, "num_steps": num_steps, "emb_size": emb_size
                             }
    dict_results["gnn_hyper_params"] = gnn_hyper_params_dict

    ### DEBUG ###
    for key in dict_results.keys():
        print(f'{key} {type(dict_results[key])}')

    logs_path = os.path.join(root_path, 'logs')
    if not os.path.exists(logs_path):
        os.mkdir(logs_path)
    model_path = KGNET_Config.trained_model_path
    create_dir([logs_path, model_path])
    with open(os.path.join(logs_path, model_name + '_log.json'), "w") as outfile:
        json.dump(dict_results, outfile)
    torch.save(model.state_dict(), os.path.join(model_path, model_name) + ".model")


    return dict_results
    # print("Total Time Sec=", (end_t - start_t).total_seconds())


if __name__ == '__main__':
    dataset_name = r'YAGO_3-10_isConnectedTo_D2H1' # mid-ddc400fac86bd520148e574f86556ecd19a9fb9ce8c18ce3ce48d274ebab3965 # Yago3-10_isConnectedTo
    root_path = os.path.join(KGNET_Config.datasets_output_path,)
    target_rel = r'isConnectedTo' #http://www.wikidata.org/entity/P101
    list_src_nodes = ['http://www.wikidata.org/entity/Q5484233',
                      'http://www.wikidata.org/entity/Q16853882',
                      'http://www.wikidata.org/entity/Q777117']
    K = 2
    modelID = r'mid-ddc400fac86bd520148e574f86556ecd19a9fb9ce8c18ce3ce48d274ebab3965.model'
    result = rgcn_lp(dataset_name,root_path,target_rel=target_rel,loadTrainedModel=0,list_src_nodes=list_src_nodes,modelID=modelID,epochs=10001,val_interval=200,hidden_channels=100)
    print(result)
# rgcn_lp(dataset_name='mid-0000100',
#         root_path=os.path.join(KGNET_Config.inference_path,),
#         loadTrainedModel=1,
#         target_rel = "https://dblp.org/rdf/schema#authoredBy",
#         list_src_nodes =  ['https://dblp.org/rec/conf/ctrsa/Rosie22',
#                             'https://dblp.org/rec/conf/ctrsa/WuX22',
#                             'https://dblp.org/rec/conf/padl/2022'],
#         K = 2,
#         modelID = ''
#         )