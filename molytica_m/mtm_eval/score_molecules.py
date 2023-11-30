from molytica_m.target_selector import target_selector_tools
from molytica_m.data_tools.graph_tools import graph_tools
from concurrent.futures import ProcessPoolExecutor
from molytica_m.data_tools import dataset_tools
from molytica_m.ml import iP_model, iPPI_model
from spektral.data import Graph
from tqdm import tqdm
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
smiles_batch = 100

smiles_added = 0
iP_batch_for_smiles = []
iPPI_batch_for_smiles = []

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
        iP_preds += iP_model.predict(dataset_tools.get_predict_loader(iP_batch_graphs, batch_size=1, epochs=1)) 

        with ProcessPoolExecutor() as executor:
            iPPI_batch_graphs = list(executor.map(get_iPPI_graph, iPPI_batch_for_smiles))
        iPPI_preds += iPPI_model.predict(dataset_tools.get_predict_loader(iPPI_batch_graphs, batch_size=1, epochs=1))

        iP_batch_for_smiles = []
        iPPI_batch_for_smiles = []
        smiles_added = 0

    smiles_scores[smiles] = smiles_score

# Correcting the indexing in the for loops
for idx, pred in enumerate(iP_preds):
    smiles = smiles_for_iP_preds[idx]
    smiles_scores[smiles] += -pred[0] / 14 * iP_tuples[2]

for idx, pred in enumerate(iPPI_preds):
    smiles = smiles_for_iPPI_preds[idx]
    smiles_scores[smiles] += -pred[0] / 14 * iPPI_tuples[3]

with open("molytica_m/mtm_eval/molecule_scores.json", "w") as file:
    json.dump(smiles_scores, file)