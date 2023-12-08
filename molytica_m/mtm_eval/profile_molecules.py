from molytica_m.target_selector import target_selector_tools
from molytica_m.data_tools.graph_tools import graph_tools
from molytica_m.data_tools import alpha_fold_tools
from concurrent.futures import ProcessPoolExecutor
from molytica_m.data_tools import dataset_tools
from molytica_m.ml import iP_model, iPPI_model
from tqdm import tqdm
import pandas as pd
import numpy as np
import json

iP_model = iP_model.get_trained_iP_model()
iPPI_model = iPPI_model.get_trained_iPPI_model()

iP_tuples, iPPI_tuples = target_selector_tools.get_latest_nodes_and_edges_evaluation()


def get_iP_graph(args):
    try:
        uniprot, smiles = args
        G = graph_tools.combine_graphs([graph_tools.get_graph_from_uniprot_id(uniprot),
                                            graph_tools.get_graph_from_smiles_string(smiles)])
        G.y = 1
        return G
    except Exception as e:
        print(e)
        return None

def get_iPPI_graph(args):
    try:
        uniprot1, uniprot2, smiles = args
        G = graph_tools.combine_graphs([graph_tools.get_graph_from_uniprot_id(uniprot1),
                                            graph_tools.get_graph_from_uniprot_id(uniprot2),
                                            graph_tools.get_graph_from_smiles_string(smiles)])
        G.y = 1
        return G
    except Exception as e:
            print(e)
            return None


smiles_scores = {}
smiles_batch = 1

smiles_added = 0
iP_batch_for_smiles = []

iP_is_not_none = []
iP_inputs = []
iP_preds = []
iP_result_tuples = []

# Tracking the smiles string for each prediction
smiles_for_iP_preds = []
smiles_for_iPPI_preds = []

smiles_list = dataset_tools.get_smiles_from_iPPI_DB()[:3]
for smiles in tqdm(smiles_list, desc="Scoring Molecules"):
    smiles_score = 0

    for uniprot_id in tqdm(alpha_fold_tools.get_alphafold_uniprot_ids(), desc="Analyzing single molecule"):
        iP_batch_for_smiles.append((uniprot_id, smiles))
        smiles_for_iP_preds.append(smiles)  # Track the smiles string

    smiles_added += 1

    if smiles_added == smiles_batch or smiles == smiles_list[-1]:
        print("Analyzing molecule")
        with ProcessPoolExecutor() as executor:
            iP_input_batch_graphs = list(tqdm(executor.map(get_iP_graph, iP_batch_for_smiles), total=len(iP_batch_for_smiles)))
        
        iP_input_batch_filtered = [x for x in iP_input_batch_graphs if x is not None]
        batch_true_false_markings = [True if x is not None else False for x in iP_input_batch_graphs]
        iP_inputs_batch_smiles_uniprot = [x for idx, x in enumerate(iP_batch_for_smiles) if [True if x is not None else False for x in iP_input_batch_graphs][idx]]
        batch_preds = [float(x[0]) for x in iP_model.predict(dataset_tools.get_predict_loader(iP_input_batch_filtered, batch_size=50, epochs=1))]

        iP_is_not_none += batch_true_false_markings
        iP_inputs += iP_inputs_batch_smiles_uniprot
        iP_preds += batch_preds
        iP_result_tuples += list(zip(iP_inputs_batch_smiles_uniprot, iP_preds))

        iP_batch_for_smiles = []
        smiles_added = 0

    smiles_scores[smiles] = smiles_score


with open("molytica_m/research/iP_result_tuples.json", "w") as file:
    json_data = {"iP_result_tuples": iP_result_tuples}
    json.dump(json_data, file)

print(iP_preds)
pred_idx = 0
for idx, graph_not_none in enumerate(iP_is_not_none):
    smiles = smiles_for_iP_preds[idx]
    preds = []
    if graph_not_none: # If there was data for this item
        preds.append(iP_preds[pred_idx])
        pred_idx += 1
    else:
        preds = None
    
    print(smiles)
    print(preds)