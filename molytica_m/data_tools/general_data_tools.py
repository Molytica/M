from molytica_m.data_tools.alpha_fold_tools import get_alphafold_uniprot_ids
from molytica_m.data_tools.graph_tools import graph_tools
from molytica_m.data_tools.graph_tools import interactome_tools
from scipy.sparse.csgraph import connected_components
from itertools import combinations_with_replacement
from scipy.sparse import csr_matrix
from tqdm import tqdm
import pandas as pd
import numpy as np
import random
import json
import os

def is_in_DLiP(uniprot_pair_list, DLiP_data):
    for DLiP_value in DLiP_data.values():
        if set(DLiP_value["proteins"]) == set(uniprot_pair_list): # If PPI pair is in DLiP, skip
            return True
    return False

def get_non_iPPIs(n, DLiP_data):
    # Uniprots and smiles from only the DLiP_keys
    uniprots = get_uniprots(DLiP_data.keys(), DLiP_data)
    af_uniprots = get_alphafold_uniprot_ids()
    smiles = get_smiles(DLiP_data.keys(), DLiP_data)

    all_combs = list(combinations_with_replacement(uniprots, 2))
    random.shuffle(all_combs)

    edge_list = interactome_tools.get_HuRI_table_as_uniprot_edge_list()

    non_iPPIs = set()

    for uniprot_pair in edge_list:
        uniprot_pair_list = list(uniprot_pair)
        # Get a random uniprot_pair (including homo pairs)
        # Get a smiles
        random_smiles = random.choice(list(smiles))

        if uniprot_pair_list[0] not in af_uniprots or uniprot_pair_list[1] not in af_uniprots:
            continue
        
        if not is_in_DLiP(uniprot_pair_list, DLiP_data): # Assume it is not an iPPI and add
            non_iPPI = (uniprot_pair_list[0], uniprot_pair_list[1], random_smiles)
            non_iPPIs.add(non_iPPI)
            if len(non_iPPIs) % 100 == 0:
                print(f"Adding non iPPI {len(non_iPPIs)}/{n}")

            if len(non_iPPIs) == n:
                break      
  
    return non_iPPIs

def get_DLiP_ids_from_nodes(train_nodes, DLiP_data):
    DLiP_ids = set()

    for uniprot_PPI_set in train_nodes:
        for uniprot_pair in list(combinations_with_replacement(uniprot_PPI_set, 2)):
            # Check if pair is in DLiP and add in that case
            for value in DLiP_data.values():
                pair = value["proteins"]
                if set(pair) == set(uniprot_pair):
                    DLiP_ids.add(value["compound_id"]) # incorrectly labeled as compound id but it is actually DLiP id
    
    return DLiP_ids

def get_smiles(DLiP_keys, DLiP_data):
    smiles = set()
    for key in DLiP_keys:
        smiles.add(DLiP_data[key]["SMILES"][0])
    return smiles

def get_uniprots(DLiP_keys, DLiP_data):
    proteins = set()
    for key in DLiP_keys:
        for prot in DLiP_data[key]["proteins"]:
            proteins.add(prot)
    return proteins

def get_PPI_molecules(DLiP_ids, DLiP_data, af_uniprots):
    PPI_molecules = {}
    id_count = 0
    added = 0
    for DLiP_id in DLiP_ids:
        # Positive iPPI, molecule inhibits interaction
        if DLiP_data[DLiP_id]["proteins"][0] in af_uniprots and DLiP_data[DLiP_id]["proteins"][1] in af_uniprots:
            iPPI = {"proteins": DLiP_data[DLiP_id]["proteins"], "molecule": DLiP_data[DLiP_id]["SMILES"][0], "iPPI": 1} # 0 for the canonical RdKit smiles
            PPI_molecules[id_count] = iPPI
            id_count += 2
            added += 1
    
    id_count = 1
    for non_iPPI in get_non_iPPIs(added, DLiP_data):
        iPPI = {"proteins": [non_iPPI[0], non_iPPI[1]], "molecule": non_iPPI[2], "iPPI": 0} # 0 for the canonical RdKit smiles
        PPI_molecules[id_count] = iPPI
        id_count += 2
    return PPI_molecules
    
def save_graphs(PPI_molecules, save_path, split_key):
    skipped = 0
    idx = -1
    for key in tqdm(sorted(list(PPI_molecules.keys())), desc="Generating and saving data", unit="c_graphs"):
        idx += 1
        try:
            iPPI = PPI_molecules[key]
            file_name = os.path.join(save_path, split_key, f'{idx}_{split_key}.h5')

            G = graph_tools.combine_graphs([graph_tools.get_graph_from_uniprot_id(iPPI['proteins'][0]),
                                        graph_tools.get_graph_from_uniprot_id(iPPI['proteins'][0]),
                                        graph_tools.get_graph_from_smiles_string(iPPI['molecule'])])
            G.y = np.array([iPPI["iPPI"]])
            graph_tools.save_graph(G, file_name)
        except:
            skipped += 1