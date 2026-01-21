SPARQL_ENDPOINT=""
GRAPH_URI=""
python SparqlMLaasService/KG_schema.py --graph-uri "$GRAPH_URI" --endpoint "$SPARQL_ENDPOINT"
python SparqlMLaasService/LLM_subgraph_sampler_rag.py
python extract_KG.py
python GNNaaS/DataTransform/TSV_TO_PYG_dataset.py
# Training GNN model
python GNNaaS/models/wise_saint.py --dataset_name DBLP_D1H1 loadTrainedModel=0
# Inferencing GNN model
python GNNaaS/models/wise_saint.py --dataset_name DBLP_D1H1 loadTrainedModel=1