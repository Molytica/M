from molytica_m.target_selector import target_selector_tools
from molytica_m.data_tools.graph_tools import graph_tools
from concurrent.futures import ProcessPoolExecutor
from molytica_m.data_tools import dataset_tools
from molytica_m.ml import iP_model, iPPI_model
from spektral.data import Graph
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
smiles_batch = 40

smiles_added = 0
iP_batch_for_smiles = []
iPPI_batch_for_smiles = []

iP_is_none = []
iPPI_is_none = []
iP_preds = []
iPPI_preds = []

# Tracking the smiles string for each prediction
smiles_for_iP_preds = []
smiles_for_iPPI_preds = []

smiles_list = dataset_tools.get_smiles_from_iPPI_DB()
for smiles in tqdm(smiles_list, desc="Scoring Molecules"):
    smiles_score = 0

    for iP_tuple in iP_tuples:
        iP_batch_for_smiles.append((iP_tuple[0], smiles))
        smiles_for_iP_preds.append(smiles)  # Track the smiles string

    for iPPI_tuple in iPPI_tuples:
        iPPI_batch_for_smiles.append((iPPI_tuple[0], iPPI_tuple[1], smiles))
        smiles_for_iPPI_preds.append(smiles)  # Track the smiles string

    smiles_added += 1

    if smiles_added == smiles_batch or smiles == smiles_list[-1]:
        with ProcessPoolExecutor() as executor:
            iP_batch_graphs = list(executor.map(get_iP_graph, iP_batch_for_smiles))
        iP_batch_filtered = [x for x in iP_batch_graphs if x is not None]
        iP_is_none += [True if x is not None else False for x in iP_batch_graphs]
        iP_preds += [float(x[0]) for x in iP_model.predict(dataset_tools.get_predict_loader(iP_batch_filtered, batch_size=1, epochs=1))]

        with ProcessPoolExecutor() as executor:
            iPPI_batch_graphs = list(executor.map(get_iPPI_graph, iPPI_batch_for_smiles))
        iPPI_batch_filtered = [x for x in iPPI_batch_graphs if x is not None]
        iPPI_is_none += [True if x is not None else False for x in iPPI_batch_graphs]
        iPPI_preds += [float(x[0]) for x in iPPI_model.predict(dataset_tools.get_predict_loader(iPPI_batch_filtered, batch_size=1, epochs=1))]

        iP_batch_for_smiles = []
        iPPI_batch_for_smiles = []
        smiles_added = 0

    smiles_scores[smiles] = smiles_score


# Correcting the indexing in the for loops
pred_idx = 0
for idx, graph in enumerate(iP_is_none):
    smiles = smiles_for_iP_preds[idx]
    if graph: # If there was data for this item
        smiles_scores[smiles] += iP_preds[pred_idx] * iP_tuples[idx % len(iP_tuples)][2]
        pred_idx += 1

pred_idx = 0
for idx, graph in enumerate(iPPI_is_none):
    smiles = smiles_for_iPPI_preds[idx]
    if graph: # If there was data for this item
        smiles_scores[smiles] += (iPPI_preds[pred_idx] - 0.5) * iPPI_tuples[idx % len(iPPI_tuples)][3]
        pred_idx += 1

with open("molytica_m/mtm_eval/molecule_scores.json", "w") as file:
    json.dump(smiles_scores, file)

data = [(idx, key, value) for idx, (key, value) in enumerate(smiles_scores.items())]

# Creating DataFrame
df = pd.DataFrame(data, columns=["ID", "Molecule", "Score"])

df.to_csv("molytica_m/mtm_eval/molecule_evaluations.csv")