import sys
import os

GMLaaS_models_path=sys.path[0].split("KGNET")[0]+"/KGNET/GNNaaS/models/rgcn"
sys.path.insert(0,GMLaaS_models_path)
GMLaaS_models_path=sys.path[0].split("KGNET")[0]+"/KGNET/GNNaaS/models"
sys.path.insert(0,GMLaaS_models_path)
sys.path.append(os.path.join(os.path.abspath(__file__).split("KGNET")[0],'KGNET'))


from Constants import *
from SparqlMLaasService.TaskSampler.TOSG_Extraction_NC import get_d1h1_TargetListquery,execute_sparql_multithreads_multifile

query = """
select ?s as ?subject ?p as ?predicate ?o as ?object
from  <https://yago-knowledge.org>
where 
{
?s ?p ?o.
}
limit ?limit
offset ?offset
"""
out_file = os.path.join(KGNET_Config.inference_path,'Yago_LP.tsv')
start_offset = 0
sparql_endpoint_url = 'http://206.12.97.2:8890/sparql/'
batch_size = 10000000
queries = [query]
thread_count = 10
execute_sparql_multithreads_multifile(start_offset=start_offset,sparql_endpoint_url=sparql_endpoint_url,batch_size=batch_size,
                                      queries=queries,out_file=out_file,threads_count=thread_count,)