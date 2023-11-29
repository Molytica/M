from molytica_m.target_selector import target_selector_tools
from molytica_m.data_tools.graph_tools import graph_tools
from molytica_m.data_tools import dataset_tools
from molytica_m.ml import iP_model, iPPI_model
from spektral.data import Graph
import numpy as np
import json

iP_model = iP_model.get_trained_iP_model()
iPPI_model = iPPI_model.get_trained_iPPI_model()

iP_tuples, iPPI_tuples = target_selector_tools.get_latest_nodes_and_edges_evaluation()

smiles_scores = {}
for smiles in dataset_tools.get_smiles_from_iPPI_DB():
    smiles_score = 0

    for iP_tuple in iP_tuples:
        G = graph_tools.combine_graphs([graph_tools.get_graph_from_uniprot_id(iP_tuple[0]),
                                        graph_tools.get_graph_from_smiles_string(smiles)])
        G.y = 1
        pred = iP_model.predict(dataset_tools.get_predict_loader([G], batch_size=1, epochs=1))
        smiles_score += -pred[0][0] / 14 * iP_tuple[1] # Divide by 14 to roughly normalize it against the iPPI values (around 0.5)
    
    
    for iPPI_tuple in iPPI_tuples:
        G = graph_tools.combine_graphs([graph_tools.get_graph_from_uniprot_id(iPPI_tuple[0]),
                                        graph_tools.get_graph_from_uniprot_id(iPPI_tuple[1]),
                                        graph_tools.get_graph_from_smiles_string(smiles)])
        G.y = 1
        pred = iPPI_model.predict(dataset_tools.get_predict_loader([G], batch_size=1, epochs=1))
        smiles_score += (pred[0][0] - 0.5) * iP_tuple[2]
    
    smiles_scores[smiles] = smiles_score

with open("molytica_m/mtm_eval/molecule_scores.json", "w") as file:
    json.dump(smiles_scores, file)