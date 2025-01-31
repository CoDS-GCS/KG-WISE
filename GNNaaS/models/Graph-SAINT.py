from copy import copy
import json
import argparse
import shutil
from tqdm import tqdm
import datetime
import torch
import torch.nn.functional as F
import sys
import os
GMLaaS_models_path=sys.path[0].split("KG-WISE")[0]+"/KG-WISE/GNNaaS/models"
sys.path.insert(0,GMLaaS_models_path)
sys.path.insert(0,os.getcwd())
from Constants import *

from torch.nn import ModuleList, Linear, ParameterDict, Parameter
from torch_sparse import SparseTensor
from torch_geometric.data import Data
from torch_geometric.loader import  GraphSAINTRandomWalkSampler,ShaDowKHopSampler

from GNNaaS.DataTransform.TSV_TO_PYG_dataset import transform_tsv_to_PYG
from torch_geometric.utils.hetero import group_hetero_graph
from torch_geometric.nn import MessagePassing
import psutil
import numpy
from collections import OrderedDict
# from SparqlMLaasService.TaskSampler.TOSG_Extraction_NC import get_d1h1_query
from RDFHandler import KGNET
import pandas as pd
import zipfile
# print("sys.path=",sys.path)
# from ogb.nodeproppred import PygNodePropPredDataset
from evaluater import Evaluator
from custome_pyg_dataset import PygNodePropPredDataset_hsh
from resource import *
from logger import Logger
import gzip
import faulthandler
faulthandler.enable()
import pickle
import warnings
import zarr
from concurrent.futures import ProcessPoolExecutor, as_completed,ThreadPoolExecutor
from SparqlMLaasService.TaskSampler.TOSG_Extraction_NC import get_d1h1_TargetListquery
torch.multiprocessing.set_sharing_strategy('file_system')
# subject_node = 'Paper'

def execute_query(batch,inference_file,kg):
    formatted_links = ','.join(['<' + link + '>' for link in batch[0]])
    query = """
    PREFIX dblp2022: <https://dblp.org/rdf/schema#> 
    PREFIX kgnet: <http://kgnet/> 
    SELECT DISTINCT ?s ?p ?o
    FROM <http://dblp.org>  
    WHERE {{  
        ?s <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> dblp2022:Publication .
        ?s dblp2022:publishedIn ?dblp_Venue . 
        ?s dblp2022:title ?Title . 
        ?s ?p ?o.
        FILTER(?s IN ({formatted_links}))
    }}
    """

    query = """
        PREFIX dblp2022: <https://dblp.org/rdf/schema#>
        PREFIX kgnet: <http://kgnet/>

        SELECT DISTINCT ?s ?p ?o
        from <http://dblp.org>
        where
        {{
        ?s <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> dblp2022:Publication .
        ?s dblp2022:publishedIn ?dblp_Venue .
        ?s dblp2022:title ?Title .
        #?s <https://dblp.org/rdf/schema#yearOfPublication> "2022".
        ?s <https://dblp.org/rdf/schema#publishedIn> ?conf .
        filter(?conf in ("AAAI","ACC","Appl. Math. Comput.","Autom.","BMC Bioinform.","Bioinform.","CDC","CVPR","CoRR","Commun. ACM","Discret. Math.","EMBC","EUSIPCO","Eur. J. Oper. Res.","Expert Syst. Appl.","GLOBECOM","HICSS","IACR Cryptol. ePrint Arch.","ICASSP","ICC","ICIP","ICRA","IECON","IEEE Access","IEEE Trans. Autom. Control.","IEEE Trans. Commun.","IEEE Trans. Geosci. Remote. Sens.","IEEE Trans. Ind. Electron.","IEEE Trans. Inf. Theory","IEEE Trans. Signal Process.","IEEE Trans. Veh. Technol.","IGARSS","IJCAI","IJCNN","INTERSPEECH","IROS","ISCAS","ISIT","Inf. Sci.","Lecture Notes in Computer Science","Multim. Tools Appl.","NeuroImage","Neurocomputing","PIMRC","Remote. Sens.","SMC","Sensors","Theor. Comput. Sci.","WCNC","WSC")) .
        ?s ?p ?o.
        FILTER(?s IN ({formatted_links}))
        }}
    
    """
    # query = """
    #     select distinct (?s as ?subject) (?p as ?predicate) (?o as ?object)
    #     from <http://wikikg-v2>
    #     where
    #     {
    #     ?s ?p ?o.
    #     }
    #     limit ?limit
    #     offset ?offset
    #  """
    query = query.format(formatted_links=formatted_links)
    subgraph_df = kg.KG_sparqlEndpoint.executeSparqlquery(query)
    # subgraph_df = kg.KG_sparqlEndpoint.execute_sparql_multithreads([query],inference_file)
    if len(subgraph_df.columns)!=3:
        print(subgraph_df)
        raise AssertionError
    subgraph_df = subgraph_df.applymap(lambda x: x.strip('"'))
    if os.path.exists(inference_file):
        subgraph_df.to_csv(inference_file, header=None, index=None, sep='\t', mode='a')
    else:
        subgraph_df.to_csv(inference_file, index=None, sep='\t', mode='a')

def fill_missing_rel (relation,dir):
    # warnings.warn('@@@@@@@ FILLING MISSING TRIPLE {} @@@@@@@'.format(relation), UserWarning)
    # dir = os.path.join(inference_relations, relation)
    os.mkdir(dir)
    edge_data = {0: ["-1"], 1: ["-1"]}
    pd.DataFrame(edge_data).to_csv(os.path.join(dir, 'edge.csv.gz'), header=None, index=None,
                                   compression='gzip')  # edge.csv
    pd.DataFrame(edge_data).to_csv(os.path.join(dir, 'edge_reltype.csv.gz'), header=None, index=None,
                                   compression='gzip')  # edge_reltype.csv
    pd.DataFrame({0: ["1"]}).to_csv(os.path.join(dir, 'num-edge-list.csv.gz'), header=None, index=None,
                                    compression='gzip')
def execute_query_v2(batch, inference_file,kg, graph_uri='http://wikikg-v2',):
    # batch = ['<' + target + '>' if '<' not in target or '>' not in target else target for target in batch]
    # query = get_d1h1_TargetListquery(graph_uri=graph_uri,target_lst=batch)
    subgraph_df = kg.KG_sparqlEndpoint.executeSparqlquery(batch)#kg.KG_sparqlEndpoint.execute_sparql_multithreads([query], inference_file)
    if len(subgraph_df.columns) != 3:
        print(subgraph_df)
        raise AssertionError
    subgraph_df = subgraph_df.applymap(lambda x: x.strip('"'))
    if os.path.exists(inference_file):
        subgraph_df.to_csv(inference_file, header=None, index=None, sep='\t', mode='a')
    else:
        subgraph_df.to_csv(inference_file, index=None, sep='\t', mode='a')

def batch_tosa_v2 (targetNodesList,inference_file,graph_uri,kg,BATCH_SIZE=2000):
    def batch_generator():
        ptr = 0
        while ptr < len(targetNodesList):
            yield targetNodesList[ptr:ptr + BATCH_SIZE ]
            ptr += BATCH_SIZE


    queries = []
    for q in batch_generator():
        target_lst = ['<' + target + '>' for target in q]
        queries.extend(get_d1h1_TargetListquery(graph_uri=graph_uri, target_lst=target_lst))
    if len(targetNodesList) > BATCH_SIZE:
        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(execute_query_v2, batch, inference_file, kg, graph_uri) for batch in
                       queries]
            for future in tqdm(futures, desc="Downloading Raw Graph", unit="subgraphs"):
                # future.result()
                pass

    else:
        [execute_query_v2(batch,inference_file,kg,graph_uri) for batch in queries]

def batch_tosa(path_target_csv,inference_file,kg,graph_uri,BATCH_SIZE=2000):
    def batch_generator():
        ptr = 0
        while ptr < len(df_targets):
            # yield df_targets.iloc[ptr:ptr + BATCH_SIZE, :] for DF
            yield df_targets[ptr:ptr + BATCH_SIZE]
            ptr += BATCH_SIZE

    df_targets = pd.read_csv(path_target_csv, header=None)[0].to_list()

    # if len(df_targets) > BATCH_SIZE:
    #     iterations = len(df_targets) // BATCH_SIZE
    #     remainder = len(df_targets) % BATCH_SIZE
    #     ptr = 0
    #     for i in tqdm(range(iterations), desc="Downloading Raw Graph", unit="subgraphs"):
    #         batch = df_targets.iloc[ptr:ptr + BATCH_SIZE, :]
    #         execute_query(batch, inference_file, kg)
    #         ptr+=BATCH_SIZE
    #
    #     if remainder:
    #         batch = df_targets.iloc[ptr:, :] # OLD IMP df_targets.iloc[ptr:ptr + remainder, :]
    #         execute_query(batch, inference_file, kg)
    # else:
    #     execute_query(df_targets,inference_file,kg)
    # queries = [ q_inst for q_inst in get_d1h1_TargetListquery(graph_uri=graph_uri,target_lst=['<'+target+'>' for target in q]) for q in batch_generator()]
    # queries = [
    #     q_inst  # This is the item being added to the list
    #     for q_inst in get_d1h1_TargetListquery(
    #         graph_uri=graph_uri,
    #         target_lst=['<' + target + '>' for target in q ]  # List comprehension inside
    #     )
    #     for q in batch_generator()  # Another list comprehension inside
    # ]
    queries = []
    for q in batch_generator():
        target_lst = ['<'+target+'>' for target in q]
        queries.extend(get_d1h1_TargetListquery(graph_uri=graph_uri,target_lst=target_lst))

    # kg.KG_sparqlEndpoint.execute_sparql_multithreads(queries, inference_file)
    """ Parallel Threading"""
    if len(df_targets) > BATCH_SIZE:
        with ProcessPoolExecutor() as executor:
            # futures = [executor.submit(execute_query, batch, inference_file, kg) for batch in batch_generator()]
            # futures = [executor.submit(execute_query_v2, batch, inference_file, kg,graph_uri) for batch in batch_generator()]
            futures = [executor.submit(execute_query_v2, batch, inference_file, kg, graph_uri) for batch in
                       queries]
            for future in tqdm(futures, desc="Downloading Raw Graph", unit="subgraphs"):
                # future.result()
                pass

    else:
        # execute_query(df_targets,inference_file,kg)
        execute_query_v2(df_targets,inference_file,kg)



def print_memory_usage():
    # print("max_mem_GB=",psutil.Process().memory_info().rss / (1024 * 1024*1024))
    # print("get_process_memory=",getrusage(RUSAGE_SELF).ru_maxrss/(1024*1024))
    print('used virtual memory GB:', psutil.virtual_memory().used / (1024.0 ** 3), " percent",
          psutil.virtual_memory().percent)

def gen_model_name(dataset_name,GNN_Method):
    # timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # return dataset_name+'_'+model_name+'_'+timestamp
    return dataset_name


def create_dir(list_paths):
    for path in list_paths:
        if not os.path.exists(path):
            os.mkdir(path)
""" Functions for Loading Mappings in Generate_inference_subgraph ()"""
def process_node(node,master_mapping,inference_mapping):
    try:
        df_master = pd.read_csv(os.path.join(master_mapping, node), dtype=str)
        df_inf = pd.read_csv(os.path.join(inference_mapping, node), dtype=str)

        intersection = pd.merge(df_master, df_inf, on='ent name', how='inner', suffixes=('_orig', '_inf'))

        return node.split('_entidx2name')[0], intersection
    except Exception as e:
        if node == 'relidx2relname.csv.gz' or node == 'labelidx2labelname.csv.gz':
            key = node.split('.csv.gz')[0]
            col_name = 'rel name' if node.startswith('rel') else 'label name'
            intersection = pd.merge(df_master, df_inf, on=col_name, how='inner', suffixes=('_orig', '_inf'))

            return key, intersection
        else:
            raise Exception(f"Unhandled Node {node}: {e}")

def process_all_nodes(nodes,master_mapping,inference_mapping):
    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(process_node, node,master_mapping,inference_mapping): node for node in nodes}
        results = {}

        for future in as_completed(futures):
            node = futures[future]
            try:
                result = future.result()
                results[result[0]] = result[1]
            except Exception as e:
                print(f'Error processing node {node}: {str(e)}')

    return results

def generate_inference_subgraph (master_ds_name,graph_uri='http://dblp.org',target_rel_uri='https://dblp.org/rdf/schema#publishedIn',targetNode_filter_statments='',sparqlEndpointURL='http://206.12.98.118:8890/sparql',output_file='inference_subG',targetNodesList=[]):
    # graph_uri = 'https://dblp2022.org' for dblp
    # graph_uri = 'http://wikikg-v2' for wikiKG
    # target_rel_uri = 'target_rel_uri='https://dblp.org/rdf/schema#publishedIn' for dblp
    # target_rel_uri = 'http://www.wikidata.org/entity/P27' for WikiKG country of human
    # target_rel_uri = 'http://www.wikidata.org/entity/P106' for WikiKG Occupation of human

    global download_end_time,target_masks,target_masks_inf
    time_ALL_start = datetime.datetime.now()
    # output_file = 'wiki_7k'


    inference_file = os.path.join(KGNET_Config.inference_path,output_file+'.tsv')

    download_start_time = datetime.datetime.now()
    if os.path.exists(inference_file):
        os.remove(inference_file)
        # pass
    if not targetNodesList:
        test_targets= os.path.join(KGNET_Config.datasets_output_path, TARGETS)
        target_df = pd.read_csv(test_targets,header=None)
        targetNodesList = target_df[0].tolist()

    kg = KGNET(sparqlEndpointURL,KG_Prefix='http://kgnet/')
    batch_tosa_v2(targetNodesList=targetNodesList,
               inference_file=inference_file,
               kg=kg,
               graph_uri=graph_uri)
    download_end_time = (datetime.datetime.now() - download_start_time).total_seconds()
    # print(f"******** DOWNLOAD_TIME : {download_end_time}")

    """ Non Batched Query execution """
    # subgraph_df = kg.KG_sparqlEndpoint.executeSparqlquery(inference_node_query)
    # subgraph_df = subgraph_df.applymap(lambda x : x.strip('"'))
    # subgraph_df.to_csv(inference_file,index=None,sep='\t')



    """ Mapping node ids !"""
    # master_ds_root = r'/home/afandi/GitRepos/KGNET/Datasets/mid-0000114_orig' #TODO: Replace with arg based directory
    master_ds_root = os.path.join(KGNET_Config.datasets_output_path,master_ds_name)
    master_mapping = os.path.join(master_ds_root,'mapping')
    master_relations = os.path.join(master_ds_root,'raw','relations')

    inference_root = os.path.join(KGNET_Config.inference_path,output_file)
    inference_mapping = os.path.join(inference_root,'mapping')
    inference_relations = os.path.join(inference_root,'raw','relations')

    if os.path.exists(inference_root):
        shutil.rmtree(inference_root)

    if os.path.exists(inference_root+'.zip'):
        os.remove(inference_root+'.zip')

    if not os.path.exists(master_ds_root):
        master_zip = master_ds_root+'.zip'
        if os.path.exists(master_zip):
            with zipfile.ZipFile(master_zip, 'r') as zip_ref:
                zip_ref.extractall(KGNET_Config.datasets_output_path)
        else:
            raise Exception ('No Complete Graph at {} '.format(master_ds_root))
    # transform_tsv_to_PYG
    # inference_transform_tsv_to_PYG
    time_infTransform_start = datetime.datetime.now()
    transform_tsv_to_PYG(dataset_name=output_file,
                                  dataset_name_csv=output_file,
                                  dataset_types=os.path.join(KGNET_Config.datasets_output_path,'dblp2022_Types.csv'), #TODO: Replace with arg based file # For DBLP /home/afandi/GitRepos/KGNET/Datasets/dblp2022_Types (rec).csv
                                  target_rel =target_rel_uri,#,publishedIn
                                  targetNodeType=None, #TODO: Parameterize
                                  output_root_path=KGNET_Config.inference_path,
                                  Header_row=None,
                                  labelNodetype = None,#'publishedIn',
                                  split_rel=None,   ### For Transform_tsv_to_PYG
                                  similar_target_rels=[],### For Transform_tsv_to_PYG
                                  inference=True### For Transform_tsv_to_PYG
                                   )
    time_infTransform_end = (datetime.datetime.now() - time_infTransform_start).total_seconds()
    print(f'TRANSFORMATION COMPLETE: {time_infTransform_end}')
    # master_mapping_files =  os.listdir(master_mapping)
    # master_mapping_csv = [csv for csv in master_mapping_files if csv.endswith('.csv')]
    target_node = [x for x in os.listdir(os.path.join(inference_root,'split')) if not (x.endswith('.csv') | x.endswith('.gz'))][0]
    time_mapLoad_start = datetime.datetime.now()
    inf_edges = os.listdir(inference_relations)
    num_master_relations = len(master_relations)
    """Load Mappings Serial"""
    # master_nid = [entry.name for entry in os.scandir(master_mapping) if entry.is_file() and entry.name.endswith('.csv.gz')]
    # inf_nid = [entry.name for entry in os.scandir(inference_mapping) if entry.is_file() and entry.name.endswith('.csv.gz')]
    #
    # common_node_types = set(inf_nid).intersection(set(master_nid))
    # mapping_dict = {}
    # for node in common_node_types:
    #     try:
    #         df_master = pd.read_csv(os.path.join(master_mapping,node),dtype=str)
    #         df_inf = pd.read_csv(os.path.join(inference_mapping,node),dtype=str,compression='gzip')
    #
    #         intersection = pd.merge(df_master,df_inf, on='ent name',how = 'inner',suffixes=('_orig','_inf'))
    #         mapping_dict[node.split('_entidx2name')[0]] = intersection
    #     except:
    #         print(f'Node {node} is causing error!')
    #         if node == 'relidx2relname.csv.gz':
    #             intersection = pd.merge(df_master, df_inf, on='rel name', how='inner', suffixes=('_orig', '_inf'))
    #             mapping_dict[node.split('.csv')[0]] = intersection
    #             print(f'Node {node} issue handled!')
    #
    #         elif node == 'labelidx2labelname.csv.gz':
    #             intersection = pd.merge(df_master, df_inf, on='label name', how='inner', suffixes=('_orig', '_inf'))
    #             mapping_dict[node.split('.csv')[0]] = intersection
    #             print(f'Node {node} issue handled!')
    #
    #         else:
    #             raise Exception (f"Unhandled Node {node}")
    #

    """ Loading Mapping Parallel"""
    #
    # Use glob for file listing
    import glob
    master_nid = glob.glob(os.path.join(master_mapping, '*.csv.gz'))
    inf_nid = glob.glob(os.path.join(inference_mapping, '*.csv.gz'))
    common_nodes = set(os.path.basename(node) for node in inf_nid).intersection(
        os.path.basename(node) for node in master_nid)
    mapping_dict = process_all_nodes(common_nodes, master_mapping, inference_mapping)
    #
    #
    """ ********************** """
    # global target_masks # To be used for filtering inference nodes from training nodes during inference
    print(mapping_dict.keys())

    if not len(targetNodesList) == len(mapping_dict[target_node]):
        df = pd.merge(pd.DataFrame({'ent name':targetNodesList}),mapping_dict[target_node],on='ent name',how='left')
        # target_masks = [int(x) for x in df['ent idx_orig'].tolist()]
        target_masks = [int(x) for x in mapping_dict[target_node]['ent idx_orig'].tolist()]
        print(f'df = {df}')
        target_masks_inf = [int(x) for x in df['ent idx_inf'].tolist()]
        del df
    else:
        target_masks = [int (x) for x in mapping_dict[target_node]['ent idx_orig'].tolist()]

    # print('Loading map done!')
    time_mapLoad_end = (datetime.datetime.now() - time_mapLoad_start).total_seconds()
    time_map_start = datetime.datetime.now()
    num_success_relations = 0
    num_failed_relations = 0
    num_missing_relations = 0
    for triple in inf_edges:
        try:
            src,rel,dst = triple.split('___')
            src_df = mapping_dict[src]
            dst_df = mapping_dict[dst]
            directory = os.path.join(inference_relations,triple)
            edge_df = pd.read_csv(os.path.join(directory,'edge.csv.gz'),header=None,dtype=str,compression='gzip')
            src_merged_df = pd.merge(edge_df,src_df,left_on=0,right_on='ent idx_inf',how='left') # how='inner'
            dst_merged_df = pd.merge(edge_df, dst_df, left_on=1, right_on='ent idx_inf', how='left') # how='inner'
            edge_df[0] = src_merged_df['ent idx_orig']
            edge_df[1] = dst_merged_df['ent idx_orig']

            edge_rel_df = pd.read_csv(os.path.join(directory,'edge_reltype.csv.gz'),header=None,dtype=str,compression='gzip')

            if edge_df.isnull().values.any():
                warnings.warn('______ DROPPING UNKNOWN ROWS IN {} ______'.format(triple), UserWarning)
                nan_rows_index = edge_df[edge_df.isnull().any(axis=1)].index
                edge_df = edge_df.dropna()
                edge_rel_df = edge_rel_df.drop(index = nan_rows_index)
                """ Modify the number of rows """
                pd.DataFrame({0: [str(len(edge_df))]}).to_csv(os.path.join(directory, 'num-edge-list.csv.gz'), header=None,index=None,compression='gzip')  # num-edge-list
            if len(edge_df) == 0:
                # edge_df = pd.DataFrame({0:["-1"],1:["-1"]}).to_csv(os.path.join(directory,'edge.csv.gz'),header=None,index=None,compression='gzip')
                edge_df = pd.DataFrame({0:["-1"],1:["-1"]})
                edge_df.to_csv(os.path.join(directory, 'edge.csv.gz'), header=None,index=None,compression='gzip')  # num-edge-list
                edge_df.to_csv(os.path.join(directory,'edge_reltype.csv.gz'),header=None,index=None,compression='gzip')
                pd.DataFrame({0: ["1"]}).to_csv(os.path.join(directory,'num-edge-list.csv.gz'),header=None,index=None,compression='gzip')
                num_failed_relations +=1
            else:
                edge_df.to_csv(os.path.join(directory,'edge.csv.gz'),header=None,index=None,compression='gzip')
                rel_df = mapping_dict['relidx2relname']
                current_rel_id  = edge_rel_df[0][0]
                edge_rel_df[0] = rel_df[rel_df['rel idx_inf']==str(current_rel_id)]['rel idx_orig'].iloc[0]
                edge_rel_df.to_csv(os.path.join(directory,'edge_reltype.csv.gz'),header=None,index=None,compression='gzip')
                num_success_relations+=1
            """ Compress to .gz """
            # relation_files = [file for file in os.listdir(os.path.join(inference_relations,triple)) if file.endswith('.csv')]
            #
            # for file in relation_files:
            #     input_file_comp = os.path.join(inference_relations,triple,file)
            #     output_file_comp = input_file_comp+'.gz'
            #
            #     if os.path.exists(output_file_comp):
            #         os.remove(output_file_comp)
            #
            #     with open (input_file_comp,'rb') as f_in, gzip.open(output_file_comp,'wb') as f_out:
            #         f_out.writelines(f_in)
        except Exception as e:
            print(e)
            warnings.warn('****** SKIPPING UNKNOWN TRIPLE {} ******'.format(triple), UserWarning)
            num_failed_relations += 1

    """ Map labels"""
    inf_label_dir = os.path.join(inference_root,'raw','node-label')
    for label in os.listdir(inf_label_dir):
            label_csv = os.path.join(inf_label_dir,label,'node-label.csv.gz')
            label_df = pd.read_csv(label_csv,header=None,dtype=str,compression='gzip')
            intersection = pd.merge(label_df, mapping_dict['labelidx2labelname'], left_on=0, right_on='label idx_inf',
                                    how='left').fillna(-1)
            missing_values = sum(intersection['label idx_orig']==-1)
            if missing_values > 0:
                warnings.warn('............. {}  MISSING LABELS out of {}, .i.e {:.2f}% .............'.format(missing_values,len(label_df),(missing_values/len(label_df))*100), UserWarning)
            intersection['label idx_orig'].to_csv(label_csv,header=None,index=None,compression='gzip')
            ### Compress to .gz
            # label_output_file = label_csv+'.gz'
            # if os.path.exists(label_output_file):
            #     os.remove(label_output_file)
            #
            # with open(label_csv, 'rb') as f_in, gzip.open(label_output_file, 'wb') as f_out:
            #     f_out.writelines(f_in)

    time_map_end = (datetime.datetime.now() - time_map_start).total_seconds()
    """ Filling missing triple types"""
    time_fill_start = datetime.datetime.now()
    missing_relations = set(os.listdir(master_relations)) - set(os.listdir(inference_relations))
    with ProcessPoolExecutor() as executor:
        # items = [(relation,os.path.join(inference_relations,relation)) for relation in missing_relations]
        futures = [executor.submit(fill_missing_rel,relation,os.path.join(inference_relations,relation)) for relation in missing_relations]
        for future in tqdm(futures,desc='Inserting Missing Relations', unit='relation'):
            pass
    num_missing_relations = len(missing_relations)

    """ Filling missing triples types serial"""
    # for relation in missing_relations:
    #     warnings.warn('@@@@@@@ FILLING MISSING TRIPLE {} @@@@@@@'.format(relation), UserWarning)
    #     dir = os.path.join(inference_relations,relation)
    #     os.mkdir(dir)
    #     edge_data = {0:["-1"],1:["-1"]}
    #     pd.DataFrame(edge_data).to_csv(os.path.join(dir,'edge.csv.gz'),header=None,index=None,compression='gzip') # edge.csv
    #     pd.DataFrame(edge_data).to_csv(os.path.join(dir, 'edge_reltype.csv.gz'), header=None, index=None,compression='gzip')  # edge_reltype.csv
    #     pd.DataFrame({0:["1"]}).to_csv(os.path.join(dir, 'num-edge-list.csv.gz'), header=None, index=None,compression='gzip') # num-edge-list
    #     num_missing_relations += 1
        # for file in os.listdir(dir):
        #     with open(os.path.join(dir,file),'rb') as f_in, gzip.open(os.path.join(dir,file+'.gz'),'wb') as f_out:
        #         f_out.writelines(f_in)
    time_fill_end = (datetime.datetime.now() - time_fill_start).total_seconds()


    """ Copying node mappings from Orig to the inf """
    # for file in os.listdir(master_mapping):
    #     file_path = os.path.join(master_mapping, file)
    #     old_file = os.path.join(inference_mapping, file)
    #     if os.path.exists(old_file):
    #         os.remove(old_file)
    #     shutil.copy(file_path, inference_mapping)

    """ Move meta-data inside 'raw' folder """
    time_move_start = datetime.datetime.now()
    master_meta_files = [file for file in os.listdir(os.path.join(master_ds_root,'raw')) if (file.endswith('.gz') )]#or file.endswith('.gz')) ]# and not file.startswith('num-node-dict')]
    for file in master_meta_files:
        dst_file = os.path.join(inference_root,'raw',file)
        if os.path.exists(dst_file):
            os.remove(dst_file)
        shutil.copyfile(os.path.join(master_ds_root,'raw',file),dst_file)

    """ Zip the final file """
    processed_dir = os.path.join(KGNET_Config.inference_path,output_file)
    if os.path.exists(processed_dir+'.zip'):
        os.remove(processed_dir+'.zip')


    tmp_dir = os.path.join(KGNET_Config.inference_path,'tmp')
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)

    os.makedirs(tmp_dir)
    tmp_parent = os.path.join(tmp_dir,output_file)
    os.mkdir(tmp_parent)
    shutil.copytree(processed_dir,tmp_parent,dirs_exist_ok = True)
    shutil.make_archive(processed_dir, 'zip', tmp_dir)

    shutil.rmtree(tmp_dir)
    time_move_end = (datetime.datetime.now() - time_move_start).total_seconds()
    time_ALL_end = (datetime.datetime.now() - time_ALL_start).total_seconds()
    print(8 * "*", ' DOWNLOAD TIME ', download_end_time, "*" * 8)
    print(8 * "*", ' TRANSFORMATION TIME ', time_infTransform_end, "*" * 8)
    print(8 * "*", ' LOAD MAPPINGS TIME ', time_mapLoad_end, "*" * 8)
    print(8 * "*", ' MAPPING TIME ', time_map_end, "*" * 8)
    print(8 * "*", ' FILLING TIME ', time_fill_end, "*" * 8)
    print(8 * "*", ' MOV AND ZIP TIME ', time_move_end, "*" * 8)
    print(8 * "*", ' TOTAL TIME', time_ALL_end, "*" * 8)
    print(8 * "*", ' TOTAL Master Relations ', num_master_relations, "*" * 8)
    print(8 * "*", ' SUCCESS RELATIONS ', num_success_relations, "*" * 8)
    print(8 * "*", ' FAILED RELATIONS ', num_failed_relations, "*" * 8)
    print(8 * "*", ' MISSING RELATIONS ', num_missing_relations, "*" * 8)
    print('*' * 8, ' Max RAM Usage: ', getrusage(RUSAGE_SELF).ru_maxrss / (1024 * 1024), ' GB')

    # sys.exit()
    return processed_dir#.replace('.zip','')

class RGCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels, num_node_types,
                 num_edge_types):
        super(RGCNConv, self).__init__(aggr='mean')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_node_types = num_node_types
        self.num_edge_types = num_edge_types

        self.rel_lins = ModuleList([
            Linear(in_channels, out_channels, bias=False)
            for _ in range(num_edge_types)
        ])

        self.root_lins = ModuleList([
            Linear(in_channels, out_channels, bias=True)
            for _ in range(num_node_types)
        ])

        self.reset_parameters()

    def reset_parameters(self):
        for lin in self.rel_lins:
            lin.reset_parameters()
        for lin in self.root_lins:
            lin.reset_parameters()

    def forward(self, x, edge_index, edge_type, node_type):
        out = x.new_zeros(x.size(0), self.out_channels)

        for i in range(self.num_edge_types):
            mask = edge_type == i
            out.add_(self.propagate(edge_index[:, mask], x=x, edge_type=i))

        for i in range(self.num_node_types):
            mask = node_type == i
            out[mask] += self.root_lins[i](x[mask])

        return out

    def message(self, x_j, edge_type: int):
        return self.rel_lins[edge_type](x_j)


class RGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout, num_nodes_dict, x_types, num_edge_types , state_dict = None):
        super(RGCN, self).__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.dropout = dropout

        node_types = list(num_nodes_dict.keys())
        num_node_types = len(node_types)

        self.num_node_types = num_node_types
        self.num_edge_types = num_edge_types

        ### Initialize Flat L2 ###
        # self.index = faiss.IndexFlatL2(self.in_channels)
        # self.emb_mapping = {}

        # Create embeddings for all node types that do not come with features.
        global WISE
        if not WISE:
            self.emb_dict = ParameterDict({
                f'{key}': Parameter(torch.Tensor(num_nodes_dict[key], in_channels))
                for key in set(node_types).difference(set(x_types))})
        else:
            self.emb_dict = {}

        #

        """ Comment ^ for better efficiency at inference"""

        I, H, O = in_channels, hidden_channels, out_channels  # noqa

        # Create `num_layers` many message passing layers.
        self.convs = ModuleList()
        self.convs.append(RGCNConv(I, H, num_node_types, num_edge_types))
        for _ in range(num_layers - 2):
            self.convs.append(RGCNConv(H, H, num_node_types, num_edge_types))
        self.convs.append(RGCNConv(H, O, self.num_node_types, num_edge_types))

        if not WISE:
            self.reset_parameters()
    def reset_parameters(self):
        for emb in self.emb_dict.values():
            torch.nn.init.xavier_uniform_(emb)
        for conv in self.convs:
            conv.reset_parameters()

    def group_input(self, x_dict, node_type, local_node_idx):
        # Create global node feature matrix.
        h = torch.zeros((node_type.size(0), self.in_channels),
                        device=node_type.device)

        for key, x in x_dict.items():
            mask = node_type == key
            if x.is_sparse:
                h[mask] = x.index_select(0,local_node_idx[mask]).to_dense()
                continue
            h[mask] = x[local_node_idx[mask]]

        for key, emb in self.emb_dict.items():
            mask = node_type == int(key)
            if emb.is_sparse:
                h[mask] = emb.index_select(0,local_node_idx[mask]).to_dense()
                continue
            h[mask] = emb[local_node_idx[mask]]


        return h

    def forward(self, x_dict, edge_index, edge_type, node_type,
                local_node_idx):

        x = self.group_input(x_dict, node_type, local_node_idx)

        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_type, node_type)
            if i != self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=0.5, training=self.training)

        return x.log_softmax(dim=-1)



    def store_emb(self,model_name,root_path=os.path.join(KGNET_Config.trained_model_path,'emb_store')):
        path = os.path.join(root_path,model_name)
        if not os.path.exists(root_path):
            os.mkdir(root_path)

        if os.path.exists(path):
            shutil.rmtree(path)
        os.mkdir(path)

        emb_store = zarr.DirectoryStore(path)
        root = zarr.group(store=emb_store)

        emb_mapping = {}
        # Iterate over node types in emb_dict
        ptr = 0
        for key, val in self.emb_dict.items():
            # Convert PyTorch tensor to a NumPy array with a compatible data type
            val_np = val.cpu().detach().numpy()

            # Create a Zarr array for each node type
            emb_array = root.create(key, shape=val_np.shape, dtype=val_np.dtype, chunks=(128, -1)) #chunks=(val_np.shape[0], -1)
            emb_array[:] = val_np  # Assign the embeddings to the Zarr array

            # Update the mapping information
            # emb_mapping[key] = {'start': 0, 'end': val_np.shape[0]}
            emb_mapping[key] = (ptr,ptr+val_np.shape[0]-1)
            ptr+=val_np.shape[0]#+1
        # Save the mapping information
        emb_mapping_path = os.path.join(path, 'index.map')
        with open(emb_mapping_path, 'wb') as f:
            pickle.dump(emb_mapping, f)


    def sampled_inference(self,model,homo_data,x_dict,local2global,subject_node,node_type,device='cpu'):
        model.eval()
        #x_dict[2] = x_dict[2][-100:]
        inference_nodes = local2global[subject_node]#[-100:]
        inference_nodes = torch.zeros_like(inference_nodes,dtype=torch.bool)
        # inference_nodes[-len_target:] = True
        global target_masks
        inference_nodes[target_masks] = True
        homo_data.inference_mask = torch.zeros(node_type.size(0),dtype=torch.bool)
        homo_data.inference_mask[local2global[subject_node][inference_nodes]] = True
        homo_data.inference_mask
        batch_size = 1024*2#len_target #// 100 # inference_nodes.shape[0]#x_dict[2].shape[0]
        # kwargs = {'batch_size': batch_size, 'num_workers': 0,}
        inference_loader = ShaDowKHopSampler(homo_data,depth=2,num_neighbors=20,
                                             node_idx=homo_data.inference_mask,
                                             batch_size=batch_size,
                                             num_workers=16,
                                             # **kwargs
                                             )


        pbar = tqdm(total=len(inference_loader))
        all_y_preds = []
        for data in inference_loader:
            data = data.to(device)
            out = model(x_dict,
                        data.edge_index,
                        data.edge_attr,
                        data.node_type,
                        data.local_node_idx)
            out = torch.index_select(out, 0, data.root_n_id)
            y_pred = out.argmax(dim=-1, keepdim=True).cpu()
            all_y_preds.append(y_pred)
            pbar.update(1)

        pbar.close()
        return torch.cat(all_y_preds,dim=0).squeeze(1)#y_pred.squeeze(1)
        # return y_pred.squeeze(1)

    def load_emb(self,model_name, root_path=os.path.join(KGNET_Config.trained_model_path,'emb_store')):

        path = os.path.join(root_path,model_name)
        path_map = os.path.join(path, 'index.map')

        # Load the mapping
        with open(path_map, 'rb') as f:
            emb_mapping = pickle.load(f)

        # Initialize an empty dictionary for embeddings
        tmp_emb_dict = {}

        # Create a Zarr store
        emb_store = zarr.DirectoryStore(path)
        root = zarr.group(store=emb_store)

        # Iterate over node types in emb_mapping
        for key, val in emb_mapping.items():
            # Load the Zarr array for each node type
            emb_array = root[key]

            # Convert Zarr array to a NumPy array and then to a PyTorch tensor
            embeddings = torch.from_numpy(numpy.array(emb_array))

            # Add the embeddings to the temporary dictionary
            tmp_emb_dict[key] = embeddings

        # Create an ordered dictionary of parameters
        self.emb_dict = ParameterDict({
            key: torch.nn.Parameter(tmp_emb_dict[key])
            for key in tmp_emb_dict.keys()
        })


    def selective_load_emb(self, set_neighbor_types,local2global,key2int,model_name,root_path=os.path.join(KGNET_Config.trained_model_path,'emb_store'),):
        # Define file paths
        path = os.path.join(root_path,model_name)
        path_emb = path#os.path.join(path, 'index.npy')
        path_map = os.path.join(path, 'index.map')

        # Load the mapping
        with open(path_map, 'rb') as f:
            self.emb_mapping = pickle.load(f)

        # Load the index
        # self.index = faiss.deserialize_index(numpy.load(path_emb))

        # Initialize an empty dictionary for embeddings
        tmp_emb_dict = OrderedDict()

        emb_store = zarr.DirectoryStore(path)
        root = zarr.group(store=emb_store)

        for src, _, dst in set_neighbor_types:         # Iterate node types in set_neighbor_types
            key = str(key2int[src]) if str(key2int[src]) in self.emb_mapping else str(key2int[dst])

            if key in self.emb_mapping:
                emb_array = root[key]
                embeddings = torch.from_numpy(numpy.array(emb_array))
                tmp_emb_dict[key] = embeddings

        self.emb_dict = ParameterDict({
            key: torch.nn.Parameter(tmp_emb_dict[key])
            for key in tmp_emb_dict.keys()
        })


    def load_Zarr_emb(self,edge_index_dict,key2int,model_name,target,root_path=os.path.join(KGNET_Config.trained_model_path,'emb_store'),**kwargs):
        def create_indices(v, emb_size):
            repeated_v = torch.tensor(v).repeat_interleave(emb_size)   # Repeat each element in v emb_size times
            repeated_range = torch.arange(emb_size).repeat(len(v))     # Create a tensor containing the range [0, 1, 2, ..., emb_size-1] repeated len(v) times
            indices = torch.stack([repeated_v, repeated_range])        # Combine the repeated_v and repeated_range tensors to form the final 2D tensor
            return indices


        def get_to_load(edge_index_dict,):
            to_load = {}
            for k, v in edge_index_dict.items():
                if v.equal(torch.tensor([[-1], [-1]])):
                    continue
                src, _, dst = k
                if key2int[src] != target:
                    to_load.setdefault(key2int[src], set()).update(v[0].numpy())

                if key2int[dst] != target:
                    to_load.setdefault(key2int[dst], set()).update(v[1].numpy())
            return to_load



        path=os.path.join(root_path,model_name)

        global time_map_end,time_load_end
        time_map_start = datetime.datetime.now()
        # to_load = global_to_local(edge_index,local2global)
        to_load = get_to_load(edge_index_dict)

        time_map_end = (datetime.datetime.now() - time_map_start).total_seconds()
        emb_store = zarr.DirectoryStore(path)
        root = zarr.group(store=emb_store)

        time_load_start = datetime.datetime.now()
        for k,v in to_load.items():
            v = list(v)
            root_k_shape = root[k].shape
            emb_array = torch.tensor(root[k][v].astype(float))
            v = create_indices(v,root_k_shape[1])
            sparse_tensor = torch.sparse.FloatTensor(v, emb_array.view(-1), torch.Size(root_k_shape)).to(torch.float32)
            self.emb_dict[str(k)] = sparse_tensor
        time_load_end = (datetime.datetime.now()-time_load_start).total_seconds()
        warnings.warn('total time for mapping : {}'.format(time_load_end))

        # x_dict_indices = create_indices(kwargs['target_masks'],root_k_shape[1])
        # x_dict_embds = torch.sparse.FloatTensor(x_dict_indices,torch.zeros((len(kwargs['target_masks']),root_k_shape[1])).view(-1), torch.Size((kwargs['num_target_nodes'],root_k_shape[1]))).to(torch.float32)
        # return  x_dict_embds


    def inference(self, x_dict, edge_index_dict, key2int):
        device = list(x_dict.values())[0].device

        x_dict = copy(x_dict)

        for key, emb in self.emb_dict.items():
            x_dict[int(key)] = emb

        adj_t_dict = {}
        for key, (row, col) in edge_index_dict.items():
            adj_t_dict[key] = SparseTensor(row=col, col=row).to(device)

        dict_relTime = {}
        loop_rootlins = []
        loop_tmp_gen = []
        loop_relins = []
        loop_add = []
        for i, conv in enumerate(self.convs):

            out_dict = {}
            time_infLoop0Start = datetime.datetime.now()
            for j, x in x_dict.items():
                out_dict[j] = conv.root_lins[j](x)
                # print(f'{i} x_dict type {x.dtype}')
            time_infLoop0End = (datetime.datetime.now() - time_infLoop0Start).total_seconds()
            loop_rootlins.append(time_infLoop0End)
            # print(f'{i} Time loop 0 = {time_infLoop0End}')

            try:
                time_infLoop1Start = datetime.datetime.now()
                for keys, adj_t in adj_t_dict.items():
                    time_infLoop1_0Start = datetime.datetime.now()
                    try:
                        src_key, target_key = keys[0], keys[-1]
                        out = out_dict[key2int[target_key]]
                        ## Generating adj x tmp
                        time_infLoop1_1Start = datetime.datetime.now()
                        tmp = adj_t.matmul(x_dict[key2int[src_key]], reduce='mean')
                        time_tmpGen = (datetime.datetime.now() - time_infLoop1_1Start).total_seconds()
                        loop_tmp_gen.append(time_tmpGen)
                        ## Passing tmp to rel_lins
                        tim_relStart = datetime.datetime.now()
                        tmp = conv.rel_lins[key2int[keys]](tmp).resize_([out.size()[0], out.size()[1]])
                        tim_relEnd = (datetime.datetime.now() - tim_relStart).total_seconds()
                        loop_relins.append(tim_relEnd)
                        ### Adding tmp to out
                        tim_addStart = datetime.datetime.now()
                        out.add_(tmp)
                        tim_addEnd = (datetime.datetime.now() - tim_addStart).total_seconds()
                        loop_add.append(tim_addEnd)
                        
                    except Exception as e:
                        print(f'WE ARE IN EXCEPTION MODE! {e}')
                        raise Exception

                    time_infLoop1_0End = (datetime.datetime.now() - time_infLoop1_0Start).total_seconds()
                    dict_relTime.setdefault(keys,[]).append((time_tmpGen,tim_relEnd,tim_addEnd,'=',time_infLoop1_0End))
                    if keys == ('Paper','cites','Object-cites'):
                        print('debug')
                time_infLoop1End = (datetime.datetime.now() - time_infLoop1Start).total_seconds()
                # print(f'{i} Time loop 1 = {time_infLoop1End}')
                # print(f'{i} AVG Loop 1_1 {sum(loop1_1)/len(loop1_1)}')
                # print(f'{i} AVG Loop 1_2 {sum(loop1_2)/len(loop1_2)}')
                # print(f'{i} AVG Loop 1_3 {sum(loop1_1)/len(loop1_3)}')
    

            except Exception as e:
                print(e)
                raise Exception
            time_infLoop2Start = datetime.datetime.now()
            if i != self.num_layers - 1:
                for j in range(self.num_node_types):
                    if j in out_dict:
                        F.relu_(out_dict[j])
            time_infLoop2End = (datetime.datetime.now() - time_infLoop2Start).total_seconds()
            print(f'{i} Time loop 2 = {time_infLoop2End}')
            
        # for i, conv in enumerate(self.convs):
        #     out_dict = {}

        #     for j, x in x_dict.items():
        #         out_dict[j] = conv.root_lins[j](x)


        #     for keys, adj_t in adj_t_dict.items():
        #             src_key, target_key = keys[0], keys[-1]
        #             out = out_dict[key2int[target_key]]
        #             tmp = adj_t.matmul(x_dict[key2int[src_key]], reduce='mean')
        #             tmp = conv.rel_lins[key2int[keys]](tmp).resize_([out.size()[0], out.size()[1]])
        #             out.add_(tmp)

        #     if i != self.num_layers - 1:
        #         for j in range(self.num_node_types):
        #             F.relu_(out_dict[j])

            x_dict = out_dict

        # pd.DataFrame(dict_relTime).T.to_csv('/shared_mnt/KGNET/logs/TOSA_EXP.csv',)
        print('Average Root Lins Loop time = ',sum(loop_rootlins)/len(loop_rootlins))
        return x_dict


    def Zarr_inference(self, x_dict, edge_index_dict, key2int):
        device = list(x_dict.values())[0].device

        x_dict = copy(x_dict)

        for key, emb in self.emb_dict.items():
            x_dict[int(key)] = emb

        adj_t_dict = {}
        for key, (row, col) in edge_index_dict.items():
            if torch.all(row == torch.tensor([-1])) and torch.all(col == torch.tensor([-1])):
                continue
            adj_t_dict[key] = SparseTensor(row=col, col=row).to(device)

        for i, conv in enumerate(self.convs):
            out_dict = {}

            for j, x in x_dict.items():
                out_dict[j] = conv.root_lins[j](x)

            try:
                for keys, adj_t in adj_t_dict.items():
                    try:
                        src_key, target_key = keys[0], keys[-1]
                        out = out_dict[key2int[target_key]]
                        tmp = adj_t.matmul(x_dict[key2int[src_key]], reduce='mean')
                        tmp = conv.rel_lins[key2int[keys]](tmp).resize_([out.size()[0], out.size()[1]])
                        out.add_(tmp)
                    except Exception as e:
                        print(f'{e} , rel:{keys}')
                        adj_t = adj_t.to_torch_sparse_coo_tensor()
                        adj_t = torch.sparse.FloatTensor(
                            indices=adj_t._indices(),  # Reuse original indices
                            values=adj_t._values(),  # Reuse original values
                            size=(adj_t.size(dim=0), x_dict[key2int[src_key]].size(dim=0))  # Specify the new size
                        )
                        tmp = adj_t.mm(x_dict[key2int[src_key]])
                        tmp = conv.rel_lins[key2int[keys]](tmp).resize_([out.size()[0], out.size()[1]]).to_sparse()
                        out.add_(tmp)


            except Exception as e:
                print(e)
                raise Exception

            if i != self.num_layers - 1:
                for j in range(self.num_node_types):
                    if j in out_dict:
                        F.relu_(out_dict[j])

            x_dict = out_dict

        return x_dict


    def n_neighbors(self,node_id, edge_index, n):
        visited = set()
        queue = [(node_id, 0)]  # Initialize BFS queue with the starting node and its hop count

        neighbors_at_n_hops = set()

        while queue:
            current_node, hops = queue.pop(0)  # Dequeue the node from the queue
            visited.add(current_node)
            if hops == n: # Check if the desired number of hops has been reached
                neighbors_at_n_hops.add(current_node)
                continue

            neighbors = edge_index[1, edge_index[0] == current_node].tolist() # Find neighbors of the current node

            for neighbor in neighbors: # Enqueue unvisited neighbors
                if neighbor not in visited:
                    queue.append((neighbor, hops + 1))

        return neighbors_at_n_hops



    def get_neighbors (self,list_target_nodes,edge_index):
        set_neighbors = set()
        for target_node in tqdm(list_target_nodes,desc="Gathering neighbor nodes"):
            mask = (edge_index[0] == target_node)
            neighbors = edge_index [1,mask]
            neighbors = neighbors.tolist()
            set_neighbors.update(neighbors)
        return set_neighbors



    # def get_neighbor_types(self,node_id,local2global,keys):
    #     for key in keys:
    #         if node_id in local2global[key]:
    #             return key

    def get_neighbor_types(self, set_neighbors, edge_index_dict, ):
        # for key in keys:
        set_types = set()
        edge_index_dict = copy(edge_index_dict)
        for node in tqdm(set_neighbors,desc="Getting types of neighbor nodes"):
            for key,val in edge_index_dict.items():
                if node in val:
                    set_types.update([key])
                    del edge_index_dict[key]
                    break
        return set_types




        # compare the id with values of key2int and retrieve the key(s)

    def get_neighbor_type_wrapper(self,args):
        x, local2global = args
        return x, self.get_neighbor_types(x, local2global)

    def inference_type_only(self, x_dict, edge_index_dict, key2int,local2global,edge_index,**kwargs):

        device = list(x_dict.values())[0].device
        x_dict = copy(x_dict)

        list_target_nodes = local2global['rec'].tolist()[:100] #TODO Replace Publication with target type

        set_neighbors = self.get_neighbors(list_target_nodes,edge_index)

        print(len(set_neighbors))

        # identify the type of these neighbors
        # set_types = set(map(self.get_neighbor_types,set_neighbors))
        # neighbor_types = [self.get_neighbor_types(x,local2global) for x in set_neighbors ]


        # Create a progress bar for the iteration
        neighbor_types = self.get_neighbor_types(set_neighbors,edge_index_dict)#set()
        print(f'Selected triples {len(neighbor_types)} out of {len(edge_index_dict)}')
        """ types based on local2global """
        # keys = [x for x in list(local2global.keys()) if isinstance(x, str)]
        # for x in tqdm(set_neighbors, desc="Processing neighbors"):
        #     neighbor_t = self.get_neighbor_types(x, local2global,keys)
        #     neighbor_types.add(neighbor_t)

        """ types based on edge_index_dict """

        # neighbor_to_remove = []
        # for x in tqdm(set_neighbors, desc="Processing neighbors"):
        #     neighbor_t = self.get_neighbor_types(x,edge_index_dict)
        #     if neighbor_t is not None:
        #         neighbor_types.update(neighbor_t)
        #         continue
        #     neighbor_to_remove.append(x)
        #     print(f'{x} will be removed from set_neighbors')

        # for node in neighbor_to_remove:
        #     set_neighbors.remove(node)


        print(neighbor_types)
        # only add these types to the x_dict

        """ for local2global based """
        # for key, emb in self.emb_dict.items():
        #     for n_key in neighbor_types:
        #         if key in str(key2int[n_key]):
        #             x_dict[int(key)] = emb
        #             break

        """ for edge_index_dict based """

        self.selective_load_emb(neighbor_types,local2global,key2int,model_name=kwargs['model_name'])
        # x_dict.update(self.emb_dict)
        for k,v in self.emb_dict.items():
            x_dict[int(k)] =v

        """ For Local2global based"""
        # for key, emb in self.emb_dict.items():
        #     for src,_,dst in neighbor_types:
        #         src,dst = key2int[src],key2int[dst]
        #
        #         if int(key) == src or key == dst:
        #             x_dict[int(key)] = emb
        #             breakd


        # perform inference as previous with this x_dict

        adj_t_dict = {}

        for key in neighbor_types:
            (row,col) = edge_index_dict[key]
            adj_t_dict[key] = SparseTensor(row=col, col=row).to(device)
        # for key, (row, col) in edge_index_dict.items():
        #     adj_t_dict[key] = SparseTensor(row=col, col=row).to(device)

        for i, conv in enumerate(self.convs):
            out_dict = {}

            for j, x in x_dict.items():
                out_dict[j] = conv.root_lins[j](x)

            for keys, adj_t in adj_t_dict.items():
                src_key, target_key = keys[0], keys[-1]
                out = out_dict[key2int[target_key]]
                tmp = adj_t.matmul(x_dict[key2int[src_key]], reduce='mean')
                tmp = conv.rel_lins[key2int[keys]](tmp).resize_([out.size()[0], out.size()[1]])
                out.add_(tmp)

            if i != self.num_layers - 1:
                # for j in range(self.num_node_types):
                #     F.relu_(out_dict[j])
                for k in out_dict.keys():
                    F.relu_(out_dict[k])

            x_dict = out_dict

        return x_dict



dic_results = {}

def graphShadowSaint(device=0,num_layers=2,hidden_channels=64,dropout=0.5,
                     lr=0.005,epochs=2,runs=1,batch_size=2000,walk_length=2,
                     num_steps=10,loadTrainedModel=0,dataset_name="DBLP-Springer-Papers",
                     root_path="../../Datasets/",output_path="./",include_reverse_edge=True,
                     n_classes=50,emb_size=128,label_mapping={},target_mapping={},modelID='',
                     model_obj = None,return_Model = False):
    def train(epoch):
        model.train()
        # tqdm.monitor_interval = 0
        # pbar = tqdm(total=args.num_steps * args.batch_size)
        pbar = tqdm(total=len(train_loader))
        pbar.set_description(f'Epoch {epoch:02d}')

        total_loss = total_examples = 0
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            out = model(x_dict, data.edge_index, data.edge_attr, data.node_type,data.local_node_idx)
            # out = out[data.train_mask]
            out = torch.index_select(out, 0, data.root_n_id)
            y = data.y.squeeze(1)
            loss = F.nll_loss(out, y)
            # print("loss=",loss)
            loss.backward()
            optimizer.step()
            num_examples = data.train_mask.sum().item()
            total_loss += loss.item() * num_examples
            total_examples += num_examples
            # pbar.update(args.batch_size)
            pbar.update(1)

        # pbar.refresh()  # force print final state
        pbar.close()
        # pbar.reset()
        return total_loss / total_examples

    @torch.no_grad()
    def test():
        model.eval()

        out = model.inference(x_dict, edge_index_dict, key2int)
        out = out[key2int[subject_node]]

        y_pred = out.argmax(dim=-1, keepdim=True).cpu()
        y_true = data.y_dict[subject_node]

        train_acc = evaluator.eval({
            'y_true': y_true[split_idx['train'][subject_node]],
            'y_pred': y_pred[split_idx['train'][subject_node]],
        })['acc']
        valid_acc = evaluator.eval({
            'y_true': y_true[split_idx['valid'][subject_node]],
            'y_pred': y_pred[split_idx['valid'][subject_node]],
        })['acc']
        test_acc = evaluator.eval({
            'y_true': y_true[split_idx['test'][subject_node]],
            'y_pred': y_pred[split_idx['test'][subject_node]],
        })['acc']
        return train_acc, valid_acc, test_acc

    init_ru_maxrss = getrusage(RUSAGE_SELF).ru_maxrss
    process_start = datetime.datetime.now()
    dataset_name = dataset_name

    GNN_datasets=[dataset_name]


    for GNN_dataset_name in GNN_datasets:
        # try:
        gsaint_start_t = datetime.datetime.now()
        ###################################Delete Folder if exist #############################
        dir_path=root_path+GNN_dataset_name

        """ TOSA Subgraph Inference"""
        if loadTrainedModel == 1 and WISE:
            dir_path = generate_inference_subgraph(master_ds_name=GNN_dataset_name) # TODO: parameterize for generality
            GNN_dataset_name = dir_path.split('/')[-1]
            root_path = KGNET_Config.inference_path
        """ *************************"""

        #         ####################
        """ BASELINE COMPLETE Graph TSV Transformation"""
        # if loadTrainedModel == 1 :#and not os.path.exists(os.path.join(KGNET_Config.datasets_output_path,GNN_dataset_name+'.zip')):
        #     transform_start = datetime.datetime.now()
        #     if os.path.exists(os.path.join(KGNET_Config.datasets_output_path,GNN_dataset_name+'.tsv')):
        #         transform_tsv_to_PYG(dataset_name=dataset_name,
        #                                        dataset_name_csv=dataset_name,
        #                                        dataset_types=r'/home/afandi/GitRepos/KGNET/Datasets/dblp2022_Types (rec).csv',
        #                                        # TODO: Replace with arg based file
        #                                        target_rel='publishedIn', # TODO: Replace with dynamic target
        #                                        split_rel="yearOfPublication",
        #                                        similar_target_rels = [],
        #                                        split_rel_train_value=2019,
        #                                        split_rel_valid_value=2020,
        #                                        Header_row = True,
        #                                        output_root_path=KGNET_Config.datasets_output_path)
        #         transform_end = (datetime.datetime.now() - transform_start).total_seconds()
        #         print(f'**** TRANSFORMATION TIME : {transform_end} s **********')
        #     else:
        #         raise Exception ('TSV file not found')
        evaluator = Evaluator(name='ogbn-mag')
        print("dataset_name=", dataset_name)
        dic_results = {}
        dic_results["GNN_Method"] = GNN_Methods.Graph_SAINT
        dic_results["to_keep_edge_idx_map"] = []
        dic_results["dataset_name"] = dataset_name
        if not model_obj:
            try:
                # shutil.rmtree(dir_path)
                print("Folder Deleted")
            except OSError as e:
                print("Error Deleting : %s : %s" % (dir_path, e.strerror))
            dataset = PygNodePropPredDataset_hsh(name=GNN_dataset_name, root=root_path, numofClasses=str(n_classes))
            print(getrusage(RUSAGE_SELF))
            start_t = datetime.datetime.now()
            data = dataset[0]
            # global subject_node
            subject_node = list(data.y_dict.keys())[0]
            if loadTrainedModel == 0:
                split_idx = dataset.get_idx_split()
            end_t = datetime.datetime.now()
            print("dataset init time=", end_t - start_t, " sec.")
            dic_results["dataset_load_time"] = (end_t - start_t).total_seconds()


            start_t = datetime.datetime.now()
            # We do not consider those attributes for now.
            data.node_year_dict = None
            data.edge_reltype_dict = None

            to_remove_rels = []


            for elem in to_remove_rels:
                data.edge_index_dict.pop(elem, None)
                data.edge_reltype.pop(elem, None)

            edge_index_dict = data.edge_index_dict
            ##############add inverse edges ###################
            if include_reverse_edge:
                key_lst = list(edge_index_dict.keys())
                for key in key_lst:
                    r, c = edge_index_dict[(key[0], key[1], key[2])]
                    edge_index_dict[(key[2], 'inv_' + key[1], key[0])] = torch.stack([c, r])


            out = group_hetero_graph(data.edge_index_dict, data.num_nodes_dict)
            edge_index, edge_type, node_type, local_node_idx, local2global, key2int = out
            """ Removing -1 -1 masks from the edge index"""
            mask = ~(edge_index == -1).any(dim=0)
            edge_index = edge_index[:,mask]

            homo_data = Data(edge_index=edge_index, edge_attr=edge_type,
                             node_type=node_type, local_node_idx=local_node_idx,
                             num_nodes=node_type.size(0))

            homo_data.y = node_type.new_full((node_type.size(0), 1), -1)
            try:
                homo_data.y[local2global[subject_node]] = data.y_dict[subject_node]
            except:
                warnings.warn('Warning! Mismatch in homo_data.y')

            if loadTrainedModel == 0:
                homo_data.train_mask = torch.zeros((node_type.size(0)), dtype=torch.bool)
                homo_data.train_mask[local2global[subject_node][split_idx['train'][subject_node]]] = True
                start_t = datetime.datetime.now()
                print("dataset.processed_dir", dataset.processed_dir)
                kwargs = {'batch_size': batch_size, 'num_workers': 64, 'persistent_workers': True}
                train_loader = ShaDowKHopSampler(homo_data, depth=2, num_neighbors=3,
                                                 node_idx=homo_data.train_mask,
                                                  **kwargs)

            start_t = datetime.datetime.now()
            # Map informations to their canonical type.
            #######################intialize random features ###############################
            global target_masks,target_masks_inf

            # if 'target_masks' not in globals():
            #     orig = 'WikiKG_hc_tosa'
            #     target_csv = os.path.join(KGNET_Config.datasets_output_path, 'wiki_targets_7k.csv')
            #     original_map = os.path.join(KGNET_Config.datasets_output_path, orig, 'mapping', #GNN_dataset_name
            #                                 subject_node + '_entidx2name.csv.gz')
            #
            #     target_df = pd.read_csv(target_csv, header=None, names=['ent name'])
            #     original_df = pd.read_csv(original_map)
            #     target_masks = pd.merge(original_df, target_df, on='ent name', how='inner', suffixes=('_orig', '_inf'))[
            #         'ent idx'].to_list()
            #     del target_csv, original_map, target_df, original_df
        else:
            subject_node = model_obj[1][0]
            key2int = model_obj[1][1]
            x_dict, edge_index_dict = model_obj[1][2],model_obj[1][3]
            data = model_obj[1][4]


        if not WISE:
            if isinstance(TARGETS,str):
                target_csv = os.path.join(KGNET_Config.datasets_output_path, TARGETS)
                target_df = pd.read_csv(target_csv, header=None, names=['ent name'], dtype=str)
                del target_csv
            if isinstance(TARGETS, list):
                target_df = pd.DataFrame({'ent name':TARGETS}, dtype=str)


            inf_map = os.path.join(dir_path, 'mapping', #GNN_dataset_name
                                        subject_node + '_entidx2name.csv.gz')


            
            inf_df = pd.read_csv(inf_map,dtype=str)
            target_masks = pd.merge(inf_df, target_df, on='ent name', how='inner', suffixes=('_orig', '_inf'))[
                'ent idx'].astype(int).to_list()
            del inf_map, target_df, inf_df

        """ DEBUGGING ONLY"""
        # target_masks = torch.tensor(pd.read_csv(os.path.join(dir_path, 'mapping', #GNN_dataset_name
        #                                 subject_node + '_entidx2name.csv.gz'))['ent idx'].values)
        """ **************"""
        if loadTrainedModel == 1:
            if WISE:
                feat = torch.sparse.FloatTensor(size=(data.num_nodes_dict[subject_node],emb_size))
            else:
                feat = torch.Tensor(data.num_nodes_dict[subject_node], emb_size)
        #
        else:
            feat = torch.Tensor(data.num_nodes_dict[subject_node], emb_size)
        if not WISE:
            torch.nn.init.xavier_uniform_(feat)

        feat_dic = {subject_node: feat}

        ################################################################
        x_dict = {}
        for key, x in feat_dic.items():
            x_dict[key2int[key]] = x

        num_nodes_dict = {}
        for key, N in data.num_nodes_dict.items():
            num_nodes_dict[key2int[key]] = N

        end_t = datetime.datetime.now()
        if not model_obj:
            print("model init time CPU=", end_t - start_t, " sec.")

        device = f'cuda:{device}' if torch.cuda.is_available() else 'cpu'


        process_end = (datetime.datetime.now() - process_start).total_seconds()
        process_ram = getrusage(RUSAGE_SELF).ru_maxrss/ (1024 * 1024)
        if loadTrainedModel == 1:
            time_start_loadModel = datetime.datetime.now()
            with torch.no_grad():
                y_true = data.y_dict[subject_node]
                global download_end_time,time_map_end, time_load_end
                trained_model_path = KGNET_Config.trained_model_path + modelID
                # model_name = gen_model_name(dataset_name, dic_results["GNN_Method"])
                model_params_path = trained_model_path.replace('.model', '.param')

                with open(model_params_path, 'rb') as f:
                    dict_model_param = pickle.load(f)

                if len(target_mapping) == 0:
                    target_mapping = pd.read_csv(os.path.join(dir_path, 'mapping', f'{subject_node}_entidx2name.csv.gz'),compression='gzip')
                    target_mapping = target_mapping.set_index('ent idx')['ent name'].to_dict()
                if PRUNE:
                    dict_model_param['hidden_channels'] = dict_model_param['hidden_channels']//2
                load_start = datetime.datetime.now()
                if not model_obj:
                    model = RGCN(dict_model_param['emb_size'],
                                 dict_model_param['hidden_channels'],
                                 dict_model_param['dataset.num_classes'],
                                 dict_model_param['num_layers'],
                                 dict_model_param['dropout'],
                                 #num_nodes_dict,
                                 dict_model_param['num_nodes_dict'],
                                 dict_model_param['list_x_dict_keys'],
                                 dict_model_param['len_edge_index_dict_keys'])

                """ ************** RW SAMPLER ****************"""
                # inf_loader = GraphSAINTRandomWalkSampler(homo_data,batch_size=batch_size,walk_length=walk_length,num_steps=num_steps,
                #                     sample_coverage=0,save_dir=dataset.processed_dir)
                # model.load_Zarr_emb(edge_index_dict, key2int, modelID.split('.model')[0], target=key2int[subject_node],
                #                     num_target_nodes=num_nodes_dict[key2int[subject_node]], target_masks=target_masks)


                """ ************ WISE Sampled Inference ****************"""
                # model.load_state_dict(torch.load(trained_model_path),strict=False)
                # if WISE:
                #     # pass
                #     model.load_Zarr_emb(edge_index_dict,key2int,modelID.split('.model')[0],target=key2int[subject_node],num_target_nodes = num_nodes_dict[key2int[subject_node]],target_masks = target_masks )

                # print(' Sampled Inference ')
                # y_pred = model.sampled_inference(model,homo_data,x_dict,local2global,subject_node,node_type,)#y_true.size()[0])
                # y_pred = y_pred.unsqueeze(1)
                # end_t = datetime.datetime.now()
                # dic_results["InferenceTime"] = (end_t - start_t).total_seconds()
                # print(f'y_true: {y_true.size()}\ny_pred: {y_pred.size()}\nTarget masks: {len(target_masks)}')

                # if y_true.size()[0] == y_pred.size()[0] and "target_masks_inf" in globals():
                #     y_true=y_true[target_masks_inf]
                #     y_pred = y_pred[target_masks_inf]
                # elif y_true.size()[0] > y_pred.size()[0]:
                #     y_true = y_true[target_masks]
                # elif y_true.size()[0] < y_pred.size()[0]:
                #     y_pred = y_pred[target_masks]

                # test_acc = evaluator.eval({
                #     'y_true': y_true,#[target_masks_inf],
                #     'y_pred': y_pred,#[target_masks_inf],
                # })['acc']

                # total_time = process_end+dic_results["InferenceTime"]
                # if 'download_end_time' in locals() or 'download_end_time' in globals():
                #     # process_end = process_end-download_end_time
                #     pass
                # else:
                #     download_end_time = 0
                # y_t_elem,y_t_freq = torch.unique(y_true, return_counts=True)
                # y_p_elem,y_p_freq = torch.unique(y_pred, return_counts=True)
                # y_true_count = dict(zip(y_t_elem.numpy(), y_t_freq.numpy()))
                # y_pred_count = dict(zip(y_p_elem.numpy(), y_p_freq.numpy()))
                # print(f'y_true count: {y_true_count}\ny_pred_count: {y_pred_count}')
                # print('*' * 8, '\tRAM USAGE BEFORE Model/Inference:\t', process_ram, ' GB')
                # print('*'*8, '\tDOWNLOAD TIME: ', download_end_time, 's')
                # print('*'*8, '\tTOTAL PROCESSING TIME: ', process_end, 's')
                # # print('*' * 8, '\tZARR MAPPING:\t\t',time_map_end, 's')
                # # print('*' * 8, '\tZARR LOADING:\t\t',time_load_end, 's')
                # print('*'*8,'\tInference TIME: ',dic_results["InferenceTime"],'s')
                # print('*'*8,'\tTotal TIME: ', total_time ,'s',)
                # print('*'*8,'\tMax RAM Usage: ',getrusage(RUSAGE_SELF).ru_maxrss/ (1024 * 1024),)
                # print('*'*8,'\tAccracy: ',test_acc,)
                # print('*'*8,'\tClasses in DS: ',len(y_true.unique()))#,'\n',y_true.value_counts())
                # return
                """ ************ ********************* ****************"""


                """ Load Zarr Embeddings (load_min_emb)"""
                if not model_obj:
                    model.load_state_dict(torch.load(trained_model_path),strict=False)

                else: # reference to model object
                    model = model_obj[0]
                print(model)
                if WISE:
                    model.load_Zarr_emb(edge_index_dict,key2int,modelID.split('.model')[0],target=key2int[subject_node],num_target_nodes =  num_nodes_dict[key2int[subject_node]],target_masks = target_masks )
                # model.load_emb(model_name=model_name,)
                # model.load_Zarr_emb(edge_index_dict,key2int,modelID.split('.model')[0],target=key2int[subject_node],num_target_nodes = num_nodes_dict[key2int[subject_node]],target_masks = target_masks )
                
                load_end = (datetime.datetime.now() - load_start).total_seconds()
                time_end_modelLoad = (datetime.datetime.now() - time_start_loadModel).total_seconds()
                print(f'Loaded Graph Saint Model! Time taken {load_end}')
                """ FOR SAVING WEIGHTS BIASES AND DECOUPLED EMBEDDINGS """
                # emb = copy(model.emb_dict)
                # model.emb_dict = None
                # torch.save([model.state_dict(),emb],trained_model_path.replace('trained_models','trained_models/wbe').replace('.model',f'_wbe.model'),_use_new_zipfile_serialization=False)
                # sys.exit()
                
                """ Inference type"""
                start_t = datetime.datetime.now()
                if WISE: # WISE Full batch inference
                    out = model.Zarr_inference(x_dict, edge_index_dict, key2int)

                else: #Default Full batch inference
                    out = model.inference(x_dict, edge_index_dict, key2int)
                
                # out = model.inference_type_only(x_dict, edge_index_dict, key2int,local2global,edge_index,model_name=model_name)
                """ ************** """

                out = out[key2int[subject_node]]
                y_pred = out.argmax(dim=-1, keepdim=True).cpu()#.flatten().tolist()
                end_t = datetime.datetime.now()
                dic_results["InferenceTime"] = (end_t - start_t).total_seconds()
                init_ru_maxrss = getrusage(RUSAGE_SELF).ru_maxrss

                # y_true = data.y_dict[subject_node]
                print(f'y_true: {y_true.size()}\ny_pred: {y_pred.size()}\nTarget masks: {len(target_masks)}')
                # if y_true.size()[0] > y_pred.size()[0]:
                #     y_true=y_true[target_masks]
                # elif y_true.size()[0] < y_pred.size()[0]:
                #     y_pred = y_pred[target_masks]
                total_time = process_end+time_end_modelLoad+dic_results["InferenceTime"]
                test_acc = evaluator.eval({
                    'y_true': y_true[target_masks],
                    'y_pred': y_pred[target_masks],
                })['acc']
                print('*' * 8, '\tRAM USAGE BEFORE Model/Inference:\t', process_ram, ' GB')
                print('*'*8, '\tPROCESSING TIME:\t',process_end, 's')
                print('*'*8, '\tModel Load TIME:\t',time_end_modelLoad, 's')
                # print('*' * 8, '\tZARR MAPPING:\t\t',time_map_end, 's')
                # print('*' * 8, '\tZARR LOADING:\t\t',time_load_end, 's')
                # print('*' * 8, '\tTOTAL MODEL LOAD:\t',load_end, 's')
                print('*'*8,'\tInference TIME:\t\t',dic_results["InferenceTime"],'s')
                print('*'*8,'\tTotal TIME:\t\t',total_time,'s',)
                print('*'*8,'\tMax RAM Usage:\t\t',getrusage(RUSAGE_SELF).ru_maxrss/ (1024 * 1024),' GB')
                print('*'*8,'\tAccuracy:\t\t',test_acc,)
                print('*'*8,'\tClasses in DS:\t\t',len(y_true.unique()),)
                dic_results['totalTime'] = total_time
                dic_results['maxMem'] = getrusage(RUSAGE_SELF).ru_maxrss/ (1024 * 1024)
                dic_results['accuracy'] = test_acc
                if return_Model:
                    return dic_results,(model,(subject_node,key2int,x_dict, edge_index_dict,data))
                return dic_results
            return dic_results
        else:
            model = RGCN(emb_size, hidden_channels, dataset.num_classes, num_layers,
                         dropout, num_nodes_dict, list(x_dict.keys()),
                         len(edge_index_dict.keys())).to(device)

            x_dict = {k: v.to(device) for k, v in x_dict.items()}
            # y_true = data.y_dict[subject_node]
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            model_loaded_ru_maxrss = getrusage(RUSAGE_SELF).ru_maxrss
            model_name = gen_model_name(dataset_name, dic_results["GNN_Method"])

            print("start test")
            logger = Logger(runs)
            test()  # Test if inference on GPU succeeds.
            total_run_t = 0
            for run in range(runs):
                start_t = datetime.datetime.now()
                model.reset_parameters()
                for epoch in range(1, 1 + epochs):
                    loss = train(epoch)
                    ##############
                    if loss == -1:
                        return 0.001
                        ##############

                    torch.cuda.empty_cache()
                    result = test()
                    logger.add_result(run, result)
                    train_acc, valid_acc, test_acc = result
                    print(f'Run: {run + 1:02d}, '
                          f'Epoch: {epoch:02d}, '
                          f'Loss: {loss:.4f}, '
                          f'Train: {100 * train_acc:.2f}%, '
                          f'Valid: {100 * valid_acc:.2f}%, '
                          f'Test: {100 * test_acc:.2f}%')
                logger.print_statistics(run)
                end_t = datetime.datetime.now()
                total_run_t = total_run_t + (end_t - start_t).total_seconds()
                print("model run ", run, " train time CPU=", end_t - start_t, " sec.")
                print(getrusage(RUSAGE_SELF))
            print('Calculating inference time')
            with torch.no_grad():
                time_inference_start = datetime.datetime.now()
                model.inference(x_dict, edge_index_dict, key2int)
            dic_results['Inference_Time'] = (datetime.datetime.now() - time_inference_start).total_seconds()

            total_run_t = (total_run_t + 0.00001) / runs
            gsaint_end_t = datetime.datetime.now()
            Highest_Train, Highest_Valid, Final_Train, Final_Test = logger.print_statistics()
            model_trained_ru_maxrss = getrusage(RUSAGE_SELF).ru_maxrss
            gnn_hyper_params_dict = {"device": device, "num_layers": num_layers, "hidden_channels": hidden_channels,
                                     "dropout": dropout, "lr": lr, "epochs": epochs, "runs": runs,
                                     "batch_size": batch_size, "walk_length": walk_length, "num_steps": num_steps,
                                     "emb_size": emb_size, "dataset_num_classes":dataset.num_classes,}
            logger = Logger(runs, gnn_hyper_params_dict)

            dic_results["gnn_hyper_params"] = gnn_hyper_params_dict
            dic_results['model_name'] = model_name
            dic_results["init_ru_maxrss"] = init_ru_maxrss
            dic_results["model_ru_maxrss"] = model_loaded_ru_maxrss
            dic_results["model_trained_ru_maxrss"] = model_trained_ru_maxrss
            dic_results["Highest_Train_Acc"] = Highest_Train.item()
            dic_results["Highest_Valid_Acc"] = Highest_Valid.item()
            dic_results["Final_Train_Acc"] = Final_Train.item()
            gsaint_Final_Test = Final_Test.item()
            dic_results["Final_Test_Acc"] = Final_Test.item()
            dic_results["Train_Runs_Count"] = runs
            dic_results["Train_Time"] = total_run_t
            dic_results["Total_Time"] = (gsaint_end_t - gsaint_start_t).total_seconds()
            dic_results["Model_Parameters_Count"]= sum(p.numel() for p in model.parameters())
            dic_results["Model_Trainable_Paramters_Count"]=sum(p.numel() for p in model.parameters() if p.requires_grad)
            ############### Model Hyper Parameters ###############
            dict_model_param = {}
            dict_model_param['emb_size'] = emb_size
            dict_model_param['hidden_channels'] = hidden_channels  # dataset.num_classes
            dict_model_param['dataset.num_classes'] = dataset.num_classes
            dict_model_param['num_layers'] = num_layers
            dict_model_param['dropout'] = dropout
            dict_model_param['num_nodes_dict'] = num_nodes_dict
            dict_model_param['list_x_dict_keys'] = list(x_dict.keys())
            dict_model_param['len_edge_index_dict_keys'] = len(edge_index_dict.keys())
            if len(label_mapping) > 0:
                dict_model_param['label_mapping'] = label_mapping
            logs_path = os.path.join(output_path)
            model_path = os.path.join(output_path)
            create_dir([logs_path,model_path])

            with open(os.path.join(logs_path, model_name +'_log.metadata'), "w") as outfile:
                json.dump(dic_results, outfile)
            model.store_emb(model_name=model_name)
            """REMOVING EMB DICT FROM MODEL"""
            model.emb_dict = None
            """**************************"""
            torch.save(model.state_dict(), os.path.join(model_path , model_name)+".model")
            with open (os.path.join(model_path , model_name)+".param",'wb') as f:
                pickle.dump(dict_model_param,f)

            dic_results["data_obj"] = data.to_dict()
        # except Exception as e:
        #     print(e)
        #     print(traceback.format_exc())
        #     print("dataset_name Exception")
        del train_loader
    return dic_results

def get_args(dataset_name,GCNP = False):
    global TARGETS
    if dataset_name=='YAGO_FM200':

        parser = argparse.ArgumentParser(description='OGBN-MAG (GraphShadowSAINT)')
        parser.add_argument('--modelID',type=str, default='YAGO_FM200.model')  #DBLP_Paper_Venue_FM_FTD_d1h1_PRIME_1000_GA_0_ShadowSaint Zarr = DBLP_d1h1_2021_Zarr_e10.model v2=DBLP_FTD_d1h1_Zarr
        parser.add_argument('--dataset_name', type=str, default="YAGO_FM200")#  DBLP_Paper_Venue_FM_FTD_d1h1_2021_2 ## YAGO_PC_D1H1_v2 # MAG_D1H1
        parser.add_argument('--targets', type=str, default=r'targets/YAGO_PC_D1H1_v2/YAGO_PC_D1H1_v2_1000_FG.csv')

    elif dataset_name=='DBLP15M_PV_FG':
        # TARGETS = r'targets/DBLP_D1H1/DBLP_D1H1_1000_FG.csv'
        parser = argparse.ArgumentParser(description='OGBN-MAG (GraphShadowSAINT)')
        parser.add_argument('--modelID', type=str,
                            default='DBLP15M_PV_FG.model')  # DBLP15M_PV_FG_P05_2 #DBLP_Paper_Venue_FM_FTD_d1h1_PRIME_1000_GA_0_ShadowSaint Zarr = DBLP_d1h1_2021_Zarr_e10.model v2=DBLP_FTD_d1h1_Zarr
        parser.add_argument('--dataset_name', type=str,
                            default="DBLP15M_PV_FG")  # DBLP_Paper_Venue_FM_FTD_d1h1_2021_2 ## YAGO_PC_D1H1_v2 # MAG_D1H1
        parser.add_argument('--targets', type=str, default=r'targets/DBLP_D1H1/DBLP_D1H1_1000_FG.csv')

    elif dataset_name == 'MAG42M_PV_FG':
        # TARGETS = r'targets/MAG_D1H1/MAG_D1H1_1000.csv'
        parser = argparse.ArgumentParser(description='OGBN-MAG (GraphShadowSAINT)')
        parser.add_argument('--modelID',type=str, default='MAG42M_PV_FG.model') #DBLP_Paper_Venue_FM_FTD_d1h1_PRIME_1000_GA_0_ShadowSaint Zarr = DBLP_d1h1_2021_Zarr_e10.model v2=DBLP_FTD_d1h1_Zarr
        parser.add_argument('--dataset_name', type=str, default="MAG42M_PV_FG")#  DBLP_Paper_Venue_FM_FTD_d1h1_2021_2 ## YAGO_PC_D1H1_v2 # MAG_D1H1
        parser.add_argument('--targets', type=str, default=r'targets/MAG_D1H1/MAG_D1H1_1000.csv')

    else:
        parser = argparse.ArgumentParser(description='OGBN-MAG (GraphShadowSAINT)')
        parser.add_argument('--dataset_name', type=str,
                            default=dataset_name)  # DBLP_Paper_Venue_FM_FTD_d1h1_2021_2 ## YAGO_PC_D1H1_v2 # MAG_D1H1
        parser.add_argument('--modelID',type=str,default=dataset_name+'.model')
        parser.add_argument('--targets', type=str, default=r'targets/YAGO_PC_D1H1_v2/YAGO_PC_D1H1_v2_1000.csv')

    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--hidden_channels', type=int, default=64)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--runs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=2000)
    parser.add_argument('--walk_length', type=int, default=2)
    parser.add_argument('--num_steps', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--loadTrainedModel', type=int, default=1)
    parser.add_argument('--root_path', type=str, default= KGNET_Config.datasets_output_path)
    parser.add_argument('--output_path', type=str, default=KGNET_Config.trained_model_path)
    parser.add_argument('--include_reverse_edge', type=bool, default=True)
    parser.add_argument('--n_classes', type=int, default=200)
    parser.add_argument('--emb_size', type=int, default=128)
    args = parser.parse_args()
    return args

    args = parser.parse_args()
    return args

def inference_baseline_API(dataset_name,targetNodesList=None,model_id='',return_Model = False,model_obj=None):
    global args,SAMPLED_INFERENCE,RW_SAMPLER,TARGETS,WISE,PRUNE
    SAMPLED_INFERENCE,RW_SAMPLER,WISE,PRUNE = False,False,False,False
    args = get_args(dataset_name)
    if targetNodesList is None:
        TARGETS = args.targets
    else:
        TARGETS = targetNodesList


    if not model_id == '':
        args.model_id = model_id

    return graphShadowSaint(args.device,args.num_layers,args.hidden_channels,args.dropout,args.lr,args.epochs,args.runs,
                            args.batch_size,args.walk_length,args.num_steps,args.loadTrainedModel,args.dataset_name,
                            args.root_path,args.output_path,args.include_reverse_edge,args.n_classes,args.emb_size,
                            modelID=args.modelID,return_Model=return_Model,model_obj=model_obj)

if __name__ == '__main__':
    global WISE,PRUNE #TARGETS
    WISE = False
    PRUNE = False
    """ YAGO"""
    # TARGETS = r'targets/YAGO_PC_D1H1_v2/YAGO_PC_D1H1_v2_1000_FG.csv'
    # parser = argparse.ArgumentParser(description='OGBN-MAG (GraphShadowSAINT)')
    # parser.add_argument('--modelID',type=str, default='YAGO_FM200.model')  #DBLP_Paper_Venue_FM_FTD_d1h1_PRIME_1000_GA_0_ShadowSaint Zarr = DBLP_d1h1_2021_Zarr_e10.model v2=DBLP_FTD_d1h1_Zarr
    # parser.add_argument('--dataset_name', type=str, default="YAGO_FM200")#  DBLP_Paper_Venue_FM_FTD_d1h1_2021_2 ## YAGO_PC_D1H1_v2 # MAG_D1H1

    """ DBLP"""
    # TARGETS = r'targets/DBLP_D1H1/DBLP_D1H1_1000_FG.csv'
    # parser = argparse.ArgumentParser(description='OGBN-MAG (GraphShadowSAINT)')
    # parser.add_argument('--modelID',type=str, default='DBLP15M_PV_FG.model')  #DBLP15M_PV_FG_P05_2 #DBLP_Paper_Venue_FM_FTD_d1h1_PRIME_1000_GA_0_ShadowSaint Zarr = DBLP_d1h1_2021_Zarr_e10.model v2=DBLP_FTD_d1h1_Zarr
    # parser.add_argument('--dataset_name', type=str, default="DBLP15M_PV_FG")#  DBLP_Paper_Venue_FM_FTD_d1h1_2021_2 ## YAGO_PC_D1H1_v2 # MAG_D1H1

    """ MAG"""
    # TARGETS = r'targets/MAG_D1H1/MAG_D1H1_1000.csv'
    # parser = argparse.ArgumentParser(description='OGBN-MAG (GraphShadowSAINT)')
    # parser.add_argument('--modelID',type=str, default='MAG_D1H1.model') #DBLP_Paper_Venue_FM_FTD_d1h1_PRIME_1000_GA_0_ShadowSaint Zarr = DBLP_d1h1_2021_Zarr_e10.model v2=DBLP_FTD_d1h1_Zarr
    # parser.add_argument('--dataset_name', type=str, default="MAG_D1H1")#  DBLP_Paper_Venue_FM_FTD_d1h1_2021_2 ## YAGO_PC_D1H1_v2 # MAG_D1H1
    
    # parser.add_argument('--device', type=int, default=0)
    # parser.add_argument('--num_layers', type=int, default=2)
    # parser.add_argument('--hidden_channels', type=int, default=64)
    # parser.add_argument('--dropout', type=float, default=0.5)
    # parser.add_argument('--lr', type=float, default=0.005)
    # parser.add_argument('--runs', type=int, default=1)
    # parser.add_argument('--batch_size', type=int, default=2000)
    # parser.add_argument('--walk_length', type=int, default=2)
    # parser.add_argument('--num_steps', type=int, default=10)
    # parser.add_argument('--epochs', type=int, default=10)
    # parser.add_argument('--loadTrainedModel', type=int, default=1)
    # parser.add_argument('--root_path', type=str, default= KGNET_Config.datasets_output_path)
    # parser.add_argument('--output_path', type=str, default=KGNET_Config.trained_model_path)
    # parser.add_argument('--include_reverse_edge', type=bool, default=True)
    # parser.add_argument('--n_classes', type=int, default=200)
    # parser.add_argument('--emb_size', type=int, default=128)

    # args = parser.parse_args('')
    # if WISE:
    #     args.modelID = args.modelID.replace('.model','_wise.model')
    # print(args)


    print( inference_baseline_API(dataset_name='YAGO_PC_D1H1_v2'))
    # print(graphShadowSaint(args.device,args.num_layers,args.hidden_channels,args.dropout,args.lr,args.epochs,args.runs,args.batch_size,args.walk_length,args.num_steps,args.loadTrainedModel,args.dataset_name,args.root_path,args.output_path,args.include_reverse_edge,args.n_classes,args.emb_size,modelID=args.modelID))



