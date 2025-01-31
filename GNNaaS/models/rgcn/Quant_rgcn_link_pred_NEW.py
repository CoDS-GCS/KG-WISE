""""
Implements the link prediction task on the FB15k237 datasets according to the
`"Modeling Relational Data with Graph Convolutional Networks"
<https://arxiv.org/abs/1703.06103>`_ paper.

Caution: This script is executed in a full-batch fashion, and therefore needs
to run on CPU (following the experimental setup in the official paper).
"""
import sys
import os

GMLaaS_models_path=sys.path[0].split("KGNET")[0]+"/KGNET/GNNaaS/models/rgcn"
sys.path.insert(0,GMLaaS_models_path)
sys.path.insert(0,sys.path[0].split("KGNET")[0]+"KGNET")
sys.path.insert(0, os.getcwd())
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))


# sys.path.insert(1,os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Constants import *
import argparse
import torch
import torch.nn.functional as F
from torch.nn import Parameter
from tqdm import tqdm
from rel_link_pred_dataset import RelLinkPredDataset
#from torch_geometric.datasets import RelLinkPredDataset
from torch_geometric.nn import GAE, RGCNConv
from resource import *
import datetime
import json
import pandas as pd
import os.path as osp
from torch_geometric.nn.conv import MessagePassing
import torch.nn as nn
from GNNaaS.models.rgcn.utils import uniform,load_data, generate_sampled_graph_and_labels, build_test_graph, calc_mrr
import numpy as np
from torch.nn import Identity,ModuleDict
# from dq.quantization import IntegerQuantizer
# from dq.multi_quant import evaluate_prob_mask
from GNNaaS.models.dq.quantization import IntegerQuantizer
from GNNaaS.models.dq.multi_quant import evaluate_prob_mask
from GNNaaS.models.dq.linear_quantized import LinearQuantized




def create_quantizer(qypte, ste, momentum, percentile, signed, sample_prop):
    if qypte == "FP32":
        return Identity
    else:
        return lambda: IntegerQuantizer(
            4 if qypte == "INT4" else 8,
            signed=signed,
            use_ste=ste,
            use_momentum=momentum,
            percentile=percentile,
            sample=sample_prop,
        )


def make_quantizers(qypte, dq, sign_input, ste, momentum, percentile, sample_prop):
    if dq:
        # GIN doesn't apply DQ to the LinearQuantize layers so we keep the
        # default inputs, weights, features keys.
        # See NOTE in the multi_quant.py file
        layer_quantizers = {
            "inputs": create_quantizer(
                qypte, ste, momentum, percentile, sign_input, sample_prop
            ),
            "weights": create_quantizer(
                qypte, ste, momentum, percentile, True, sample_prop
            ),
            "features": create_quantizer(
                qypte, ste, momentum, percentile, True, sample_prop
            ),
        }
        mp_quantizers = {
            # "message_low": create_quantizer(
            #     qypte, ste, momentum, percentile, True, sample_prop
            # ),
            # "message_high": create_quantizer(
            #     "FP32", ste, momentum, percentile, True, sample_prop
            # ),
            # "update_low": create_quantizer(
            #     qypte, ste, momentum, percentile, True, sample_prop
            # ),
            # "update_high": create_quantizer(
            #     "FP32", ste, momentum, percentile, True, sample_prop
            # ),
            # "aggregate_low": create_quantizer(
            #     qypte, ste, momentum, percentile, True, sample_prop
            # ),
            # "aggregate_high": create_quantizer(
            #     "FP32", ste, momentum, percentile, True, sample_prop
            # ),
        }
    else:
        layer_quantizers = {
            "inputs": create_quantizer(
                qypte, ste, momentum, percentile, sign_input, sample_prop
            ),
            "weights": create_quantizer(
                qypte, ste, momentum, percentile, True, sample_prop
            ),
            "features": create_quantizer(
                qypte, ste, momentum, percentile, True, sample_prop
            ),
        }
        mp_quantizers = {
            "message": create_quantizer(
                qypte, ste, momentum, percentile, True, sample_prop
            ),
            "update_q": create_quantizer(
                qypte, ste, momentum, percentile, True, sample_prop
            ),
            "aggregate": create_quantizer(
                qypte, ste, momentum, percentile, True, sample_prop
            ),
        }
    return layer_quantizers, mp_quantizers




def gen_model_name(dataset_name='',GNN_Method=''):
    # timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # return dataset_name+'_'+model_name+'_'+timestamp
    return dataset_name
def create_dir(list_paths):
    for path in list_paths:
        if not os.path.exists(path):
            os.mkdir(path)



class RGCN(torch.nn.Module):
    def __init__(self, num_entities, num_relations, num_bases, dropout,dim_size = 100, qypte='INT4', dq=True, ste=False, momentum=False,
                 percentile=0.001, sample_prop=None):
        super(RGCN, self).__init__()

        self.entity_embedding = nn.Embedding(num_entities, dim_size)
        self.relation_embedding = nn.Parameter(torch.Tensor(num_relations, dim_size))

        nn.init.xavier_uniform_(self.relation_embedding, gain=nn.init.calculate_gain('relu'))
        lq_signed, _ = make_quantizers(
            qypte,
            dq,
            True,
            ste=ste,
            momentum=momentum,
            percentile=percentile,
            sample_prop=sample_prop,
        )
        self.conv1 = RGCNConvLin(
            dim_size, dim_size, num_relations * 2,  layer_quantizers=lq_signed)
        self.conv2 = RGCNConvLin(
            dim_size, dim_size, num_relations * 2, layer_quantizers=lq_signed)

        self.dropout_ratio = dropout



    def forward(self, entity, edge_index, edge_type, edge_norm):
        x = self.entity_embedding(entity)
        x = F.relu(self.conv1(x, edge_index, edge_type, edge_norm))
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        x = self.conv2(x, edge_index, edge_type, edge_norm)

        return x

    def distmult(self, embedding, triplets):
        s = embedding[triplets[:, 0]]
        r = self.relation_embedding[triplets[:, 1]]
        o = embedding[triplets[:, 2]]
        score = torch.sum(s * r * o, dim=1)

        return score

    def score_loss(self, embedding, triplets, target):
        score = self.distmult(embedding, triplets)

        return F.binary_cross_entropy_with_logits(score, target)

    def reg_loss(self, embedding):
        return torch.mean(embedding.pow(2)) + torch.mean(self.relation_embedding.pow(2))


class RGCNConvLin(MessagePassing):

    def __init__(self, in_channels, out_channels, num_relations,root_weight=True,layer_quantizers=None, edge_quantizers=None,
                  **kwargs):
        super(RGCNConvLin, self).__init__(aggr='mean', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_relations = num_relations


        self.relation_linear = nn.ModuleList([
            LinearQuantized(in_channels, out_channels, bias=False,layer_quantizers = layer_quantizers) for _ in range(num_relations)
        ])
        self.root_linear = LinearQuantized(in_channels, out_channels, bias=False,layer_quantizers = layer_quantizers) if root_weight else None

        # self.layer_quant_fns = layer_quantizers
        # self.edge_quant_fns = layer_quantizers

        self.reset_parameters()
        if args.inference:
            self.deQuant = torch.ao.quantization.DeQuantStub()
            self.quant = torch.ao.quantization.QuantStub()

    def reset_parameters(self):
        # for rel_layer in self.relation_linear:
        #     nn.init.xavier_uniform_(rel_layer.weight)
        for lin in self.relation_linear:
            lin.reset_parameters()
        self.root_linear.reset_parameters()
        # if self.edge_quant_fns is not None:
        #     self.edge_quantizers = ModuleDict({
        #         str(key): ModuleDict({str(k): self.edge_quant_fns[k]() for k in self.edge_quant_fns.keys()})
        #         for key in range(self.num_relations)
        #     })
        # size = self.num_bases * self.in_channels
        # uniform(size, self.basis)
        # uniform(size, self.att)
        # uniform(size, self.root)
        # uniform(size, self.bias)

    def forward(self, x, edge_index, edge_type, edge_norm=None, size=None):
        """"""
        return self.propagate(edge_index, size=size, x=x, edge_type=edge_type,
                              edge_norm=edge_norm)
    def message(self, x_j, edge_type):
        # Apply relation-specific linear transformation
        out = torch.zeros_like(x_j)
        for i, rel_layer in enumerate(self.relation_linear):
            mask = (edge_type == i)  # Find the edges of this relation type
            if len(x_j[mask]) > 0:
                out[mask] = rel_layer(x_j[mask])
        return out

    # def message(self, x_j, edge_index_j, edge_type, edge_norm):
    #     out = torch.zeros_like(x_j)
    #     types = edge_type
    #     for i, rel_layer in enumerate(self.relation_linear):
    #         mask = (edge_type == i)  # Find the edges of this relation type
    #         if not args.inference:
    #             if self.edge_quantizers is not None:
    #                 # for type in edge_type[0].unique(),edge_type[1]
    #                 Qx_j = self.edge_quantizers[edge_type].inputs(x_j[mask])
    #                 Qw = self.edge_quantizers[edge_type].weights(rel_layer.weight)
    #                 out[mask] = torch.matmul(Qx_j, Qw.T)
    #             else:
    #                 out[mask] = rel_layer(x_j[mask])
    #
    #
    #         else: # IF INFERENCE
    #             x_j = self.deQuant(x_j)
    #             if self.edge_quantizers is not None:
    #
    #                 Qx_j = self.edge_quantizers[edge_type].inputs(x_j[mask])
    #                 Qw = rel_layer._weight_bias()[0]
    #                 torch.matmul(Qx_j, Qw.dequantize().T)
    #             else:
    #                 out[mask] = rel_layer(x_j[mask])
    #     return out

    def update(self, aggr_out, x):
        if self.root_linear is not None and x is not None:
            aggr_out = aggr_out + self.root_linear(x)
        return aggr_out

    def __repr__(self):
        return '{}({}, {}, num_relations={})'.format(
            self.__class__.__name__, self.in_channels, self.out_channels,
            self.num_relations)



def negative_sampling(edge_index, num_nodes):
    # Sample edges by corrupting either the subject or the object of each edge.
    mask_1 = torch.rand(edge_index.size(1)) < 0.5
    mask_2 = ~mask_1

    neg_edge_index = edge_index.clone()
    neg_edge_index[0, mask_1] = torch.randint(num_nodes, (mask_1.sum(), ))
    neg_edge_index[1, mask_2] = torch.randint(num_nodes, (mask_2.sum(), ))
    return neg_edge_index

def train(train_triplets, model, use_cuda, batch_size, split_size, negative_sample, reg_ratio, num_entities, num_relations):

    train_data = generate_sampled_graph_and_labels(train_triplets, batch_size, split_size, num_entities, num_relations, negative_sample)

    if use_cuda:
        device = torch.device('cuda')
        train_data.to(device)

    entity_embedding = model(train_data.entity, train_data.edge_index, train_data.edge_type, train_data.edge_norm)
    loss = model.score_loss(entity_embedding, train_data.samples, train_data.labels) + reg_ratio * model.reg_loss(entity_embedding)

    return loss

@torch.no_grad()
def compute_rank(ranks):
    # fair ranking prediction as the average
    # of optimistic and pessimistic ranking
    true = ranks[0]
    optimistic = (ranks > true).sum() + 1
    pessimistic = (ranks >= true).sum()
    return (optimistic + pessimistic).float() * 0.5


def valid(valid_triplets, model, test_graph, all_triplets):

    entity_embedding = model(test_graph.entity, test_graph.edge_index, test_graph.edge_type, test_graph.edge_norm)
    mrr,hits = calc_mrr(entity_embedding, model.relation_embedding, valid_triplets, all_triplets, hits=[1, 3, 10])

    return mrr,hits

def test(test_triplets, model, test_graph, all_triplets):

    entity_embedding = model(test_graph.entity, test_graph.edge_index, test_graph.edge_type, test_graph.edge_norm)
    mrr,hits = calc_mrr(entity_embedding, model.relation_embedding, test_triplets, all_triplets, hits=[1, 3, 10])

    return mrr,hits

def get_test_emb(test_triplets, model, test_graph, all_triplets):

    entity_embedding = model(test_graph.entity, test_graph.edge_index, test_graph.edge_type, test_graph.edge_norm)
    return entity_embedding

def rgcn_lp(dataset_name,
            root_path=KGNET_Config.datasets_output_path,
            epochs=3,val_interval=2,
            hidden_channels=0,batch_size=-1,runs=1,
            emb_size=128,walk_length = 2, num_steps=2,
            loadTrainedModel=0,
            target_rel = '',
            list_src_nodes = [],
            K = 1,modelID = ''
            ):

    init_ru_maxrss = getrusage(RUSAGE_SELF).ru_maxrss
    dict_results = {}
    if os.path.exists(os.path.join(root_path,dataset_name,'data.pt')):
        print('removing cached dataset...')
        os.remove(os.path.join(root_path,dataset_name,'data.pt'))
        os.remove(os.path.join(root_path,dataset_name,'pre_filter.pt'))
        os.path.join(root_path, dataset_name, 'pre_transform.pt')
        print('transformed dataset cleared!')
    print('loading dataset..')

    n_bases = 4 #TODO
    dropout = 0.3 #TODO
    use_cuda = False #TODO
    graph_batch_size = 20000#30000
    hidden_channels=100
    negative_sample = 1
    regularization = 1e-2
    grad_norm = 1.0
    graph_split_size = 0.5
    best_mrr = 0


    start_data_t = datetime.datetime.now()
    # dataset = RelLinkPredDataset(root_path, dataset_name)
    # global data,model,optimizer
    # data = dataset[0]
    entity2id, relation2id, train_triplets, valid_triplets, test_triplets = load_data(os.path.join(root_path,dataset_name))
    all_triplets = torch.LongTensor(np.concatenate((train_triplets, valid_triplets, test_triplets)))
    test_graph = build_test_graph(len(entity2id), len(relation2id), train_triplets)
    valid_triplets = torch.LongTensor(valid_triplets)
    test_triplets = torch.LongTensor(test_triplets)
    load_data_t = str((datetime.datetime.now()-start_data_t).total_seconds())
    dict_results['Sampling_Time'] = load_data_t
    print(f'dataset loaded at {load_data_t}')

    print('Initializing model...')
    # model = GAE(
    #     RGCNEncoder(data.num_nodes, hidden_channels=hidden_channels,
    #                 num_relations=dataset.num_relations),
    #     DistMultDecoder(dataset.num_relations // 2, hidden_channels=hidden_channels),
    # )

    model = RGCN(len(entity2id), len(relation2id), num_bases=n_bases, dropout=dropout,dim_size=hidden_channels)

    # backend = "x86"
    # for _, mod in model.named_modules():
    #     if isinstance(mod, torch.nn.Embedding):
    #         mod.qconfig = torch.ao.quantization.float_qparams_weight_only_qconfig
    #     else:
    #         mod.qconfig = torch.quantization.get_default_qconfig(backend)
    # torch.backends.quantized.engine = backend
    # torch.quantization.prepare(model, inplace=True)
    # torch.quantization.convert(model, inplace=True)


    print('Model Initialized!')
    print(model)

    """ save model as quantized """
    if loadTrainedModel == 2:
        trained_model_path = os.path.join(KGNET_Config.trained_model_path,modelID)
        model.load_state_dict(torch.load(trained_model_path)); print(f'LOADED PRE-TRAINED MODEL {modelID}')
        backend = "x86"
        for _, mod in model.named_modules():
            if isinstance(mod, torch.nn.Embedding):
                mod.qconfig = torch.ao.quantization.float_qparams_weight_only_qconfig
            else:
                mod.qconfig = torch.quantization.get_default_qconfig(backend)
        torch.backends.quantized.engine = backend
        torch.quantization.prepare(model, inplace=True)
        torch.quantization.convert(model, inplace=True)
        torch.save(model.state_dict(),trained_model_path)
        sys.exit()

    if loadTrainedModel == 1:
        mapped_src = [entity2id[j] for j in list_src_nodes]
        inf_triplets = []
        for i in [test_triplets,valid_triplets]:
            for k in i:
                if k[0].item() in (mapped_src):
                    inf_triplets.append(k)
        inf_triplets = inf_triplets[:len(list_src_nodes)]

        backend = "x86"
        for _, mod in model.named_modules():
            if isinstance(mod, torch.nn.Embedding):
                mod.qconfig = torch.ao.quantization.float_qparams_weight_only_qconfig
            else:
                mod.qconfig = torch.quantization.get_default_qconfig(backend)
        torch.backends.quantized.engine = backend
        torch.quantization.prepare(model, inplace=True)
        torch.quantization.convert(model, inplace=True)

        trained_model_path = os.path.join(KGNET_Config.trained_model_path,modelID)
        model.load_state_dict(torch.load(trained_model_path)); print(f'LOADED PRE-TRAINED MODEL {modelID}')

        """ For Saving Partial Model"""
        z = get_test_emb(inf_triplets, model, test_graph, all_triplets)
        print(f' NUMBER OF PARAMETERS = {sum(p.numel() for p in model.parameters())}')
        torch.save([model.state_dict(),z], trained_model_path.replace('trained_models','trained_models/wbe').replace('.model', f'_INF_{len(list_src_nodes)}.model'))
        sys.exit()
        """ *********************"""

        test_mrr, hits_test = test(inf_triplets, model, test_graph, all_triplets)
        return test_mrr, hits_test

        # return y_pred



    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    model_loaded_ru_maxrss = getrusage(RUSAGE_SELF).ru_maxrss
    start_train_t = datetime.datetime.now()
    best_test_mrr = 0
    best_valid_mrr = 0
    best_test_hits = 0
    best_valid_hits = 0
    tqdm_obj = tqdm (range(1, epochs),desc='Training Epochs',position=0, leave=True)
    for epoch in  (tqdm_obj):
        # print('Starting training .. ')
        model.train()
        optimizer.zero_grad()
        loss = train(train_triplets, model, use_cuda, batch_size=graph_batch_size,
                     split_size=graph_split_size,
                     negative_sample=negative_sample, reg_ratio=regularization, num_entities=len(entity2id),
                     num_relations=len(relation2id))

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_norm)
        optimizer.step()
        tqdm_obj.set_postfix_str(f"Train Loss: {loss:.4f} at epoch {epoch:05d}")
        
        if (epoch % val_interval) == 0:
            if use_cuda:
                model.cpu()

            model.eval()
            valid_mrr,hits_valid = valid(valid_triplets, model, test_graph, all_triplets)
            test_mrr,hits_test = test(test_triplets, model, test_graph, all_triplets)
            print(f'loss = {loss:.4f}')
            if valid_mrr > best_valid_mrr:
                best_mrr = valid_mrr
                # torch.save({'state_dict': model.state_dict(), 'epoch': epoch},
                #            f'best_valid_mrr_model_{dataset_name}_{target_rel}.pth')

            if test_mrr > best_test_mrr:
                best_test_mrr = test_mrr
                # torch.save({'state_dict': model.state_dict(), 'epoch': epoch},
                #            f'best_valid_mrr_model_{dataset_name}_{target_rel}.pth')

            if hits_test[10] > best_test_hits:
                best_test_hits = hits_test[10]

            if hits_valid[10] > best_valid_hits:
                best_valid_hits = hits_valid[10]

            if use_cuda:
                model.cuda()
            print(getrusage(RUSAGE_SELF))
    model_trained_ru_maxrss = getrusage(RUSAGE_SELF).ru_maxrss
    dict_results['Results'] = {'Best_MRR' : best_mrr,'Best_Hits@10' : best_test_hits}
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
    dict_results["Highest_Test_MRR"] = best_test_mrr#.item()
    dict_results["Highest_Valid_MRR"] = best_valid_mrr#.item()
    dict_results["Highest_Test_Hits@10"] = str(best_test_hits)
    dict_results["Highest_Valid_Hits@10"] = str(best_valid_hits)
    dict_results["Final_Test_MRR"] = test_mrr#.item()
    dict_results["Final_Valid_MRR"] = valid_mrr#.item()
    dict_results["Final_Test_Hits@10"] = str(hits_test[10])#str(test_hits_10)
    dict_results["Final_Valid_Hits@10"] =str(hits_valid[10]) #str(valid_hits_10)
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
    torch.save(model.state_dict(), os.path.join(model_path, model_name) + "DQ-LP.model")


    return dict_results
    # print("Total Time Sec=", (end_t - start_t).total_seconds())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Quantization RGCN Link-Prediction')
    parser.add_argument('--inference', type=bool, default=False)
    args = parser.parse_args()
    inference = 1
    dataset_name = "mid-ddc400fac86bd520148e574f86556ecd19a9fb9ce8c18ce3ce48d274ebab3965"#"dblp_LP"#"YAGO3-10_LP_D1H1"#"KGTOSA_YAGO3-10"###  WikiKG2_LP_P106_D2H1 # mid-ddc400fac86bd520148e574f86556ecd19a9fb9ce8c18ce3ce48d274ebab3965
    root_path = os.path.join(KGNET_Config.datasets_output_path,)
    target_rel = r'http://www.yago3-10/isConnectedTo'# YAGO3-10'isConnectedTo' #http://www.wikidata.org/entity/P101
    # list_src_nodes = ['http://www.wikidata.org/entity/Q5484233',
    #                   'http://www.wikidata.org/entity/Q16853882',
    #                   'http://www.wikidata.org/entity/Q777117']
    NUM_TARGETS = 1600
    list_src_nodes = pd.read_csv(os.path.join(KGNET_Config.datasets_output_path,dataset_name,'test.txt'),header=None,sep='\t')[0].to_list()[:NUM_TARGETS]
    K = 2
    modelID ="mid-ddc400fac86bd520148e574f86556ecd19a9fb9ce8c18ce3ce48d274ebab3965DQ-LP.model"#"dblp_LPDQ-LP.model"#"YAGO3-10_LP_D1H1DQ-LP.model"#"KGTOSA_YAGO3-10DQ-LP.model"## #
    epochs = 2001
    val_interval = 500
    all_time = []
    all_hits_10 = []
    runs = 3
    if inference:
        for _ in range(runs):
            time_start = datetime.datetime.now()
            result = rgcn_lp(dataset_name,root_path,target_rel=target_rel,loadTrainedModel=inference,list_src_nodes=list_src_nodes,modelID=modelID,epochs=epochs,val_interval=val_interval)
            time_end = (datetime.datetime.now() - time_start).total_seconds()
            hits_10 = result[1][10]
            all_time.append(time_end)
            all_hits_10.append(hits_10)
            print(f'MAX RAM USAGE = {getrusage(RUSAGE_SELF).ru_maxrss / (1024 * 1024)}')

    else:
        rgcn_lp(dataset_name, root_path, target_rel=target_rel, loadTrainedModel=inference,
                list_src_nodes=list_src_nodes, modelID=modelID, epochs=epochs, val_interval=val_interval)
    avg_time = sum(all_time) / len(all_time)
    avg_hits_10 = sum(all_hits_10) / len(all_hits_10)
    print('*' * 18)
    print(f'Average Hits @ 10 = {avg_hits_10}')
    print(f'Average run time = {avg_time}')
    print(f'MAX RAM USAGE = {getrusage(RUSAGE_SELF).ru_maxrss / (1024 * 1024)}')
    print('*' * 18)
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