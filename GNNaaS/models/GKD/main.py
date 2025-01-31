import sys
import os

GMLaaS_models_path=sys.path[0].split("KGNET")[0]+"/KGNET/GNNaaS/models/rgcn"
sys.path.insert(0,GMLaaS_models_path)
sys.path.insert(0,sys.path[0].split("KGNET")[0]+"KGNET")
sys.path.insert(0, os.getcwd())
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))

from Constants import *
import argparse
import os, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from torch.utils import data
from torch_geometric.utils import to_undirected, subgraph, add_remaining_self_loops, add_self_loops
from torch_scatter import scatter
from torch_geometric.data import Data
from torch_geometric.loader import GraphSAINTRandomWalkSampler
from logger import Logger, SimpleLogger
from dataset import load_nc_dataset
from data_utils import normalize, gen_normalized_adjs, evaluate, eval_acc, eval_rocauc, eval_f1, to_sparse_tensor, \
    load_fixed_splits, remove_edges
from parse import parse_method, parser_add_main_args
from torch_geometric.utils.hetero import group_hetero_graph
import copy
from GNNaaS.models.custome_pyg_dataset import PygNodePropPredDataset_hsh
torch.autograd.set_detect_anomaly(True)
from models import GeoDist
from parse import parser_add_main_args
from torch_sparse import SparseTensor
from tqdm import tqdm
from datetime import datetime
from resource import *
import pandas as pd
import shutil
# NOTE: data splits are consistent given fixed seed, see data_utils.rand_train_test_idx
def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

### Parse args ###
# parser = argparse.ArgumentParser(description='General Training Pipeline')
# parser_add_main_args(parser)
# args = parser.parse_args()
# print(args)
#
#
#
#
# fix_seed(args.seed)
#

""" ARGS """

parser = argparse.ArgumentParser(description='OGBN-MAG (GraphShadowSAINT)')
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--num_layers', type=int, default=2)
parser.add_argument('--hidden_channels', type=int, default=64)
parser.add_argument('--dropout', type=float, default=0.2)
parser.add_argument('--lr', type=float, default=0.05)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--runs', type=int, default=1)
parser.add_argument('--batch_size', type=int, default=None) #147277
parser.add_argument('--walk_length', type=int, default=2)
parser.add_argument('--num_steps', type=int, default=10)
parser.add_argument('--loadTrainedModel', type=int, default=1)
parser.add_argument('--dataset_name', type=str,default="YAGO310_Person-Aff_50_FG")
parser.add_argument('--root_path', type=str, default=KGNET_Config.datasets_output_path)
parser.add_argument('--output_path', type=str, default="./")
parser.add_argument('--include_reverse_edge', type=bool, default=True)
parser.add_argument('--n_classes', type=int, default=50)
parser.add_argument('--emb_size', type=int, default=128)
parser.add_argument('--modelID', type=str,default=None) # 'ogbn_mag_GDK_T.model'
parser.add_argument('--inference', type=str,default=True)
TARGETS = os.path.join(os.path.join(KGNET_Config.datasets_output_path,'targets','YAGO310_Person-Aff_50_FG/YAGO310_Person-Aff_50_FG_800.csv'))
parser_add_main_args(parser)
args = parser.parse_args()

"""  """
# if args.cpu:
#     device = torch.device("cpu")
# else:
#     device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

device = 'cpu'
print(device)

### Load and preprocess data ###
#dataset = load_nc_dataset(args.dataset, args.sub_dataset, args.data_dir)
time_start_process = datetime.now()
GNN_dataset_name = args.dataset_name
root_path = KGNET_Config.datasets_output_path
dir_path = os.path.join(root_path,GNN_dataset_name)
try:
    shutil.rmtree(dir_path)
    print("Folder Deleted")
except OSError as e:
    print("Error Deleting : %s : %s" % (dir_path, e.strerror))

n_classes = args.n_classes
dataset = PygNodePropPredDataset_hsh(name=GNN_dataset_name, root=root_path, numofClasses=str(n_classes))
data = dataset[0]
subject_node = list(data.y_dict.keys())[0]
edge_index_dict = data.edge_index_dict

data.label = data.y_dict[subject_node]
key_lst = list(edge_index_dict.keys())
# FOR G-SAINT
# for key in key_lst:
#     r, c = edge_index_dict[(key[0], key[1], key[2])]
#     edge_index_dict[(key[2], 'inv_' + key[1], key[0])] = torch.stack([c, r])

split_idx = dataset.get_idx_split()
out = group_hetero_graph(data.edge_index_dict, data.num_nodes_dict)
edge_index, edge_type, node_type, local_node_idx, local2global, key2int = out

""" RGCN"""
# Convert to new transposed `SparseTensor` format and add reverse edges.
data.adj_t_dict = {}
total_size = 0
for keys, (row, col) in data.edge_index_dict.items():
    sizes = (data.num_nodes_dict[keys[0]], data.num_nodes_dict[keys[2]])
    total_size += (data.num_nodes_dict[keys[0]] * data.num_nodes_dict[keys[2]])
    adj = SparseTensor(row=row, col=col, sparse_sizes=sizes)
    data.adj_t_dict[(keys[0].replace(".", "_"), keys[1].replace(".", "_"), keys[2].replace(".", "_"))] = adj.t()
    data.adj_t_dict[(keys[2].replace(".", "_"), '_inv_' + keys[1].replace(".", "_"), keys[0].replace(".", "_"))] = adj



test_adj_t = {} 
for keys, (row, col) in data.edge_index_dict.items():
    # type_edge_index = data.adj_t_dict[k]
    num = int(row.shape[0] * (1 - args.priv_ratio))
    idx = torch.randperm(row.shape[0])[:num]

    sizes = (data.num_nodes_dict[keys[0]], data.num_nodes_dict[keys[2]])
    adj = SparseTensor(row=row[idx], col=col[idx], sparse_sizes=sizes)
    # edge_index_share = type_edge_index[idx]  # [:, idx]
    test_adj_t[(keys[0].replace(".", "_"), keys[1].replace(".", "_"), keys[2].replace(".", "_"))] = adj.t()
    test_adj_t[(keys[2].replace(".", "_"), '_inv_' + keys[1].replace(".", "_"), keys[0].replace(".", "_"))] = adj

# test_adj_t = data.adj_t_dict#copy.copy(data.edge_index_dict)
# data.edge_index_dict = None

### For G-SAINT ###
# homo_data = Data(edge_index=edge_index, edge_attr=edge_type,node_type=node_type,
#                  local_node_idx=local_node_idx,num_nodes=node_type.size(0))
#
# homo_data.y = node_type.new_full((node_type.size(0), 1), -1)
# homo_data.y[local2global[subject_node]] = data.y_dict[subject_node]
# homo_data.train_mask = torch.zeros((node_type.size(0)), dtype=torch.bool)
# homo_data.train_mask[local2global[subject_node][split_idx['train'][subject_node]]] = True

# train_loader = GraphSAINTRandomWalkSampler(
#     homo_data,
#     batch_size=args.batch_size,
#     walk_length=args.walk_length,
#     num_steps=args.num_steps,
#     sample_coverage=0,
#     save_dir=dataset.processed_dir)
### ########## ###

if len(data.label.shape) == 1:
    data.label = data.label.unsqueeze(1)
num_nodes = node_type.size(0)
print(data.label.shape)
data.label = data.label.to(device)

# get the splits for all runs
# if args.rand_split:
#     split_idx_lst = [dataset.get_idx_split(train_prop=args.train_prop, valid_prop=args.valid_prop)
#                      for _ in range(args.runs)]
# elif args.dataset in ['ogbn-proteins', 'ogbn-arxiv', 'ogbn-products']:
#     split_idx_lst = [dataset.load_fixed_splits()
#                      for _ in range(args.runs)]
# else:
#     split_idx_lst = load_fixed_splits(dataset, name=args.dataset, protocol=args.protocol)



# if args.dataset == 'ogbn-proteins':
#     if args.method == 'mlp' or args.method == 'cs':
#         dataset.graph['node_feat'] = scatter(dataset.graph['edge_feat'], dataset.graph['edge_index'][0],
#                                              dim=0, dim_size=dataset.graph['num_nodes'], reduce='mean')
#     else:
#         dataset.graph['edge_index'] = to_sparse_tensor(dataset.graph['edge_index'],
#                                                        dataset.graph['edge_feat'], dataset.graph['num_nodes'])
#         dataset.graph['node_feat'] = dataset.graph['edge_index'].mean(dim=1)
#         dataset.graph['edge_index'].set_value_(None)
#     dataset.graph['edge_feat'] = None

feat = torch.Tensor(data.num_nodes_dict[subject_node], args.emb_size)
torch.nn.init.xavier_uniform_(feat)
feat_dic = {subject_node: feat}
x_dict = {}
for key, x in feat_dic.items():
    x_dict[str(key2int[key])] = x

n = num_nodes#dataset.graph['num_nodes']
# infer the number of classes for non one-hot and one-hot labels
c = max(data.label.max().item() + 1, data.label.shape[1])
d = x_dict[str(key2int[key])].shape[1]#dataset.graph['node_feat'].shape[1]

# whether or not to symmetrize
# if not args.directed and args.dataset != 'ogbn-proteins':
#     dataset.graph['edge_index'] = to_undirected(dataset.graph['edge_index'])
# edge_index_directed = dataset.graph['edge_index'][:, dataset.graph['edge_index'][1,:] >= dataset.graph['edge_index'][0,:] ]
# edge_index_directed = edge_index_directed.to(device)


print(f"num nodes {n} | num classes {c} | num node feats {d}")
num_nodes_dict = {}
for key, N in data.num_nodes_dict.items():
    num_nodes_dict[str(key2int[key])] = N

time_end_process = (datetime.now() - time_start_process).total_seconds()
print(f'Time for pre-process = {time_end_process}')
### Load method ###
# model = parse_method(args, dataset, n, c, d, device)
time_model_init_start = datetime.now()
model = GeoDist(in_channels= args.emb_size,hidden_channels = args.hidden_channels,
                out_channels = dataset.num_classes,args=args,num_node = num_nodes,
                hetero_params=[num_nodes_dict,str(key2int[subject_node]),list(data.adj_t_dict.keys())])# edge_index_dict
time_model_init_end = (datetime.now() - time_model_init_start).total_seconds()
# using rocauc as the eval function
if args.dataset in ('yelp-chi', 'deezer-europe', 'twitch-e', 'fb100', 'ogbn-proteins'):
    criterion = nn.BCEWithLogitsLoss()
else:
    criterion = nn.NLLLoss()
if args.metric == 'rocauc':
    eval_func = eval_rocauc
elif args.metric == 'f1':
    eval_func = eval_f1
else:
    eval_func = eval_acc

logger = Logger(args.runs, args)
model.train()
print('MODEL:', model)
# dataset.graph['edge_index'], dataset.graph['node_feat'] = \
#     dataset.graph['edge_index'].to(device), dataset.graph['node_feat'].to(device)
data.graph = {}
data.graph['edge_index'], data.graph['node_feat'] =  edge_index.to(device), x_dict[str(key2int[subject_node])].to(device)

if args.dataset in ('yelp-chi', 'deezer-europe', 'fb100', 'twitch-e', 'ogbn-proteins'):
    if dataset.label.shape[1] == 1:
        dataset.label = F.one_hot(dataset.label, dataset.label.max() + 1).squeeze(1)

dataset_mask = copy.deepcopy(data)

### Training loop ###

for run in range(args.runs):
    # if args.dataset in ['cora', 'citeseer', 'pubmed'] and args.protocol == 'semi':
    #     split_idx = split_idx_lst[0]
    # else:
    #     split_idx = split_idx_lst[run]
    train_idx = split_idx['train'][subject_node].to(device)
    dataset.train_idx = train_idx

    # Processing for privileged information in each run
    if args.priv_type == 'edge':
        pass
        # test_adj_t = {}
        # for k in data.adj_t_dict.keys():
        #     type_edge_index = data.adj_t_dict[k]
        #     num = int(type_edge_index.size(1) * (1 - args.priv_ratio))
        #     idx = torch.randperm(type_edge_index.size(1))[:num]
        #     edge_index_share = type_edge_index[idx]#[:, idx]
        #     test_adj_t[k] = edge_index_share

            # row_indices = type_edge_index.indices()[0]
            # col_indices = type_edge_index.indices()[1]
            #
            # num_links = row_indices.size(0) # Get the number of non-zero entries (links)
            # random_indices = torch.randperm(num_links)[:num]
            # # Use the random indices to select rows and columns
            # row_selected = row_indices[random_indices]
            # col_selected = col_indices[random_indices]
            # new_edge_index = SparseTensor(row=row_selected, col=col_selected,
            #                               sparse_sizes=(type_edge_index.size(0), type_edge_index.size(1)))
            # test_adj_t[k] = new_edge_index
        """ DROPPING TRAINING ROWS FOR SCALABILITY """
        random_idx = torch.randperm(train_idx.shape[0])#[:args.batch_size]
        y = data.y_dict[subject_node].squeeze(1)#[split_idx['train'][subject_node]]
        df = pd.Series(y)
        y_counts = df[split_idx['train'][subject_node].tolist()].value_counts().to_dict()
        share_node_idx = []
        for label in df.unique():
            if label == -1: # ignoring targets without labels
                continue
            if args.batch_size is None:
                threshold = int(y_counts[label] * args.priv_ratio)
            else:
                threshold =  int(y_counts[label] * (args.batch_size / y.shape[0]))#int((y_counts[label] * args.priv_ratio) * (args.batch_size/c))
                threshold = threshold if threshold > 0 else 1
            label_idx = torch.nonzero((y==label)[split_idx['train'][subject_node]]).squeeze(1).tolist()
            share_node_idx.extend(label_idx[:threshold])
        share_node_idx = torch.tensor(share_node_idx)
        # train_idx = train_idx[random_idx][:args.batch_size]
        # share_node_idx = torch.cat([train_idx],dim=-1)

    elif args.priv_type == 'node':
        mask_subg = {}
        for k,v in data.adj_t_dict.items():
            train_num = data.num_nodes_dict[k[0]].shape[0]
            num = int((1 - args.priv_ratio) * train_num) # removing certain ratio of train nodes on training node
            assert num < train_num
            share_train_idx = train_idx[torch.randperm(train_num)[:num]]
            share_node_idx = torch.cat([share_train_idx, split_idx['valid'][subject_node].to(device), split_idx['test'][subject_node].to(device)], dim=-1)
            #dataset_mask.graph['edge_index'] = subgraph(share_node_idx, data.graph['edge_index'])[0]
            dataset_mask.graph['edge_index'] = subgraph(local2global[subject_node][share_node_idx], data.graph['edge_index'])[0]
            dataset_mask.train_idx = local2global[subject_node][share_train_idx]
            dataset_mask.share_node_idx = local2global[subject_node][share_node_idx]

        #prev
        train_num = train_idx.shape[0]
        num = int((1 - args.priv_ratio) * train_num) # removing certain ratio of train nodes on training node
        assert num < train_num
        share_train_idx = train_idx[torch.randperm(train_num)[:num]]
        share_node_idx = torch.cat([share_train_idx, split_idx['valid'][subject_node].to(device), split_idx['test'][subject_node].to(device)], dim=-1)
        #dataset_mask.graph['edge_index'] = subgraph(share_node_idx, data.graph['edge_index'])[0]
        dataset_mask.graph['edge_index'] = subgraph(local2global[subject_node][share_node_idx], data.graph['edge_index'])[0]
        dataset_mask.train_idx = local2global[subject_node][share_train_idx]
        dataset_mask.share_node_idx = local2global[subject_node][share_node_idx]

        ###
        # mask_homo_data = Data(edge_index=dataset_mask.graph['edge_index'], edge_attr=edge_type,
        #                  node_type=node_type, local_node_idx=local_node_idx,
        #                  num_nodes=node_type.size(0))
        #
        # mask_homo_data.y = node_type.new_full((node_type.size(0), 1), -1)
        # mask_homo_data.y[local2global[subject_node]] = data.y_dict[subject_node]
        # mask_homo_data.train_mask = torch.zeros((node_type.size(0)), dtype=torch.bool)
        # mask_homo_data.train_mask[local2global[subject_node]][share_node_idx]=True#[split_idx['train'][subject_node]]] = True
        #
        # student_loader = GraphSAINTRandomWalkSampler(mask_homo_data,
        #                                             batch_size=args.batch_size,
        #                                             walk_length=args.walk_length,
        #                                             num_steps=args.num_steps,
        #                                             sample_coverage=0,
        #                                             save_dir=dataset.processed_dir)



    else:
        raise NotImplementedError
    
    model.reset_parameters()

    if args.mode == 'train': # loading teacher model
        # model_dir = f'./saved_models/{args.base_model}_{args.dataset}_{run}.pkl'
        if args.modelID is None:
            args.modelID = f'{args.dataset_name}_GDK_T.model'
        model_dir = os.path.join(KGNET_Config.trained_model_path,args.modelID)
        if not os.path.exists(model_dir):
            raise FileNotFoundError
        else:
            model_dict = torch.load(model_dir)
            if not args.not_load_teacher:
                # pass
                model.teacher_gnn.load_state_dict(model_dict)
            if args.inference:
                std_modelID = args.modelID.replace('_GDK_T.model','_GDK_S.model')
                model_dir = os.path.join(KGNET_Config.trained_model_path, args.modelID)
                model_dict = torch.load(model_dir)
                model.student_gnn.load_state_dict(model_dict)

    """ INFERENCE"""
    if args.inference:

        result = evaluate(model, [x_dict, (test_adj_t, share_node_idx), key2int], split_idx, eval_func, criterion, args,
                          y_true=data.label, subject_node=subject_node,return_pred=True)
        print(result)
        target_csv = os.path.join(KGNET_Config.datasets_output_path, TARGETS)
        target_df = pd.read_csv(target_csv, header=None, names=['ent name'], dtype=str)
        inf_map = os.path.join(dir_path, 'mapping',subject_node + '_entidx2name.csv.gz')
        inf_df = pd.read_csv(inf_map, dtype=str)
        target_masks = pd.merge(inf_df, target_df, on='ent name', how='inner', suffixes=('_orig', '_inf'))[
            'ent idx'].astype(int).to_list()
        y_pred = result[-1][str(key2int[subject_node])].argmax(dim=-1, keepdim=True).cpu()#.flatten().tolist()
        from GNNaaS.models.evaluater import Evaluator
        evaluator = Evaluator(name=f'{GNN_dataset_name}')
        test_acc = evaluator.eval({
            'y_true': data.label[target_masks],
            'y_pred': y_pred[target_masks],
        })['acc']
        total_time = (datetime.now() - time_start_process).total_seconds()
        print(test_acc)
        # total_time = time_end_process + time_end_modelLoad + dic_results["InferenceTime"]
        # print('*' * 8, '\tPROCESSING TIME:\t', time_end_process, 's')
        # print('*' * 8, '\tModel Load TIME:\t', time_end_modelLoad, 's')
        # print('*' * 8, '\tInference TIME:\t\t', dic_results["InferenceTime"], 's')
        print('*' * 8, '\tTotal TIME:\t\t', total_time, 's', )
        print('*' * 8, '\tMax RAM Usage:\t\t', getrusage(RUSAGE_SELF).ru_maxrss / (1024 * 1024), ' GB')
        print('*' * 8, '\tAccuracy:\t\t', test_acc, )
        # print('*' * 8, '\tClasses in DS:\t\t', len(y_true.unique()), )






        sys.exit()
    """ ######################"""
    optimizer_te = torch.optim.Adam([{'params': model.teacher_gnn.parameters()}], lr=args.lr, ) # weight_decay=args.weight_decay
    optimizer_st = torch.optim.Adam([{'params': model.student_gnn.parameters()}], lr=args.lr,) #  weight_decay=args.weight_decay
    if args.dist_mode == 'pgkd': optimizer_k = torch.optim.Adam([{'params': model.k.parameters()}], lr=args.lr2, weight_decay=args.weight_decay)

    best_val = float('-inf')
    times_epoch = []
    for epoch in tqdm(range(args.epochs)):
        time_start_epoch = datetime.now()
        model.train()
        train_start = time.time()
        if args.mode == 'pretrain':
            optimizer_te.zero_grad()

            # out = model(dataset, mode='pretrain')
            # for G-SAINT sampler
            # for subg in train_loader:
                # out = model([x_dict,subg],mode='pretrain')
                # loss = criterion(out[subg.train_mask], subg.y[subg.train_mask].squeeze())
                # loss.backward()
                # optimizer_te.step()
            out = model([x_dict,data.adj_t_dict,key2int],mode='pretrain')
            out = out[str(key2int[subject_node])].log_softmax(dim=-1)
            loss = F.nll_loss(out[split_idx['train'][subject_node]],
                       data.y_dict[subject_node][split_idx['train'][subject_node]].squeeze())
            # loss = F.nll_loss(out, data.y_dict[subject_node][split_idx['train'][subject_node]].squeeze())
            loss.backward()
            optimizer_te.step()
            time_end_epoch = (datetime.now() - time_start_epoch).total_seconds()
            times_epoch.append(time_end_epoch)
            print(f'Time taken for {args.mode} epoch = {time_end_epoch}')

        elif args.mode == 'train' and args.dist_mode != 'pgkd':

            optimizer_st.zero_grad()
            # outputs = model(dataset, dataset_mask, mode='train', dist_mode=args.dist_mode, t=args.t)
            # for subg in train_loader:
            #outputs = model([x_dict[str(key2int[subject_node])],train_loader],data=[test_adj_t,train_idx,share_node_idx],mode='train',dist_mode=args.dist_mode, t=args.t) G-SAINT
            outputs = model([x_dict, data.adj_t_dict,subject_node,key2int],
                            data=[test_adj_t, train_idx, share_node_idx], mode='train', dist_mode=args.dist_mode,
                            t=args.t)
            out = outputs[0] if type(outputs) == tuple else outputs

            if args.dataset in ('yelp-chi', 'deezer-europe', 'fb100', 'twitch-e', 'ogbn-proteins'):
                sup_loss = criterion(out[dataset_mask.train_idx], dataset_mask.label.squeeze(1)[dataset_mask.train_idx].to(torch.float))
            else:
                out = F.log_softmax(out, dim=1)
                sup_loss = criterion(out[train_idx], data.label.squeeze(1)[train_idx])

            if args.dist_mode == 'no': loss = sup_loss
            elif args.dist_mode == 'gkd' and not args.use_kd:
                loss = (1 - args.alpha) * sup_loss + args.alpha * outputs[1]
            elif args.dist_mode == 'gkd' and args.use_kd:
                loss = (1 - args.alpha) * sup_loss + args.alpha * outputs[1] + args.beta * outputs[2] * args.tau * args.tau


            loss.backward()
            optimizer_st.step()
            time_end_epoch = (datetime.now() - time_start_epoch).total_seconds()
            times_epoch.append(time_end_epoch)
            print(f'Time taken for {args.mode} epoch = {time_end_epoch}')
            
        elif args.mode == 'train' and args.dist_mode == 'pgkd':
            raise NotImplementedError
            # outputs = model(dataset, dataset_mask, mode='train', dist_mode=args.dist_mode, t=args.t)
            # out = outputs[0] if type(outputs) == tuple else outputs
            #
            # if args.dataset in ('yelp-chi', 'deezer-europe', 'fb100', 'twitch-e', 'ogbn-proteins'):
            #     sup_loss = criterion(out[dataset_mask.train_idx], dataset_mask.label.squeeze(1)[dataset_mask.train_idx].to(torch.float))
            # else:
            #     out = F.log_softmax(out, dim=1)
            #     sup_loss = criterion(out[dataset_mask.train_idx], dataset_mask.label.squeeze(1)[dataset_mask.train_idx])
            #
            # if not args.use_kd:
            #     loss = (1 - args.alpha) * sup_loss + args.alpha * outputs[1]
            # else:
            #     loss = (1 - args.alpha) * sup_loss + args.alpha * outputs[1] + args.beta * outputs[3] * args.tau * args.tau
            #
            # optimizer_k.zero_grad()
            # rec_loss = outputs[2]
            # rec_loss.backward(retain_graph=True)
            # optimizer_k.step()
            #
            # optimizer_st.zero_grad()
            # loss.backward()
            # optimizer_st.step()
            
        train_time = time.time() - train_start


        if args.mode == 'pretrain':
            if args.oracle:
                result = evaluate(model, dataset, split_idx, eval_func, criterion, args, test_dataset=dataset) 
            else:
                result = evaluate(model, [x_dict,data.adj_t_dict,key2int], split_idx, eval_func, criterion, args,
                                  test_dataset=test_adj_t,y_true = data.label,subject_node = subject_node) # test_dataset=dataset_mask
        elif args.mode == 'train':
            result = evaluate(model, [x_dict,(test_adj_t,share_node_idx),key2int], split_idx, eval_func, criterion, args,y_true = data.label,subject_node=subject_node)

        
        logger.add_result(run, result[:-1])
        if result[1] > best_val:
            best_val = result[1]
            if args.dataset != 'ogbn-proteins':
                best_out = F.softmax(result[-1][key2int[subject_node]], dim=1)
            else:
                best_out = result[-1]
            # if args.mode == 'pretrain' and args.save_model:
            #     torch.save(model.teacher_gnn.state_dict(), f'./saved_models/{args.base_model}_{args.dataset}_{run}.pkl')
            #     print(f"model saved at { f'./saved_models/{args.base_model}_{args.dataset}_{run}.pkl'}")
        if epoch % args.display_step == 0:
            print(f'Epoch: {epoch:02d}, '
                  f'Loss: {loss:.4f}, '
                  f'Train: {100 * result[0]:.2f}%, '
                  f'Valid: {100 * result[1]:.2f}%, '
                  f'Test: {100 * result[2]:.2f}%')
            if args.print_prop:
                pred = out.argmax(dim=-1, keepdim=True)
                print("Predicted proportions:", pred.unique(return_counts=True)[1].float() / pred.shape[0])
    
    results = logger.print_statistics(run)

results = logger.print_statistics()
print(f'average epoch time  = {sum(times_epoch) / len(times_epoch)}')
print(f'max memory usage = {getrusage(RUSAGE_SELF).ru_maxrss / (1024**2)} GB(s)')
### Save teacher model
if args.mode == 'pretrain':
    teacher_save_path = os.path.join(KGNET_Config.trained_model_path,f'{args.dataset_name}_GDK_T.model')
    torch.save(model.teacher_gnn.state_dict(), teacher_save_path)
    print(f"Teacher model saved at {teacher_save_path} ")
### Save student model
if args.mode == 'train':
    student_save_path = os.path.join(KGNET_Config.trained_model_path,f'{args.dataset_name}_GDK_S.model')
    torch.save(model.student_gnn.state_dict(),student_save_path)
    print(f"Student model saved at {student_save_path} ")


# ### Save results ###
filename = f'./logs/{args.dataset}_{args.priv_type}.csv'
print(f"Saving results to {filename}")
with open(f"{filename}", 'a+') as write_obj:
    sub_dataset = f'{args.sub_dataset},' if args.sub_dataset else ''
    write_obj.write(f"data({args.dataset},{args.priv_type}{args.priv_ratio}), model({args.log_name},{args.base_model},{args.dist_mode}),\
        \t lr({args.lr}), wd({args.weight_decay}), alpha({args.alpha}), t({args.t}), dt({args.delta}) \t")
    write_obj.write("perf: {} $\pm$ {}\n".format(format(results.mean(), '.2f'), format(results.std(), '.2f')))