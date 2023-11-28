from molytica_m.data_tools.alpha_fold_tools import get_alphafold_uniprot_ids
from molytica_m.data_tools.graph_tools import graph_tools
from scipy.sparse.csgraph import connected_components
from itertools import combinations_with_replacement
from scipy.sparse import csr_matrix
from tqdm import tqdm
import pandas as pd
import numpy as np
import random
import json
import os


def uniprot_list_from_ensg(ensg, df_idmappings):
    uniprots = list(df_idmappings.loc[df_idmappings["From"] == ensg]["Entry"])
    return uniprots

def get_HuRI_table_as_uniprot_edge_list():
    if os.path.exists("molytica_m/data_tools/HuRI_edge_list.json"):
        print("Loading saved edgelist.")
        with open("molytica_m/data_tools/HuRI_edge_list.json", "r") as file:
            return json.load(file)["HuRI_edge_list"]
    else:
        print("Generating edgelist...")
        edge_list = []
        df = pd.read_table("molytica_m/data_tools/HuRI.tsv")
        df_idmappings = pd.read_table("molytica_m/data_tools/idmapping_2023_11_18.tsv")

        for row in tqdm(df.iterrows(), total=df.shape[0], desc="Reading mappings"):
            interact_A = row['A']
            interact_B = row['B']

            uniprots_A = uniprot_list_from_ensg(interact_A, df_idmappings)
            uniprots_B = uniprot_list_from_ensg(interact_B, df_idmappings)

            for uniprot_A in uniprots_A:
                for uniprot_B in uniprots_B:
                    edge_list.append([uniprot_A, uniprot_B])

        with open("molytica_m/data_tools/HuRI_edge_list.json", "w") as file:
            json_data = {"HuRI_edge_list": edge_list}
            json.dump(json_data, file)
        return get_HuRI_table_as_uniprot_edge_list()

def is_in_DLiP(uniprot_pair_list, DLiP_data):
    for DLiP_value in DLiP_data.values():
        if set(DLiP_value["proteins"]) == set(uniprot_pair_list): # If PPI pair is in DLiP, skip
            return True
    return False

def in_HuRI(uniprot_pair_list, edge_list):
    if uniprot_pair_list in edge_list or [uniprot_pair_list[1], uniprot_pair_list[0]] in edge_list:
        print("in HuRI")
        return True
    else:
        print("not in HuRI")
        return False

def get_non_iPPIs(n, DLiP_data):
    # Uniprots and smiles from only the DLiP_keys
    uniprots = get_uniprots(DLiP_data.keys(), DLiP_data)
    af_uniprots = get_alphafold_uniprot_ids()
    smiles = get_smiles(DLiP_data.keys(), DLiP_data)

    all_combs = list(combinations_with_replacement(uniprots, 2))
    random.shuffle(all_combs)

    edge_list = get_HuRI_table_as_uniprot_edge_list()

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

def get_subgraph_nodes_from_edgelist(edge_list):
    # Create a set of all nodes
    nodes = set(node for edge in edge_list for node in edge)
    # Create a mapping from node names to integers
    node_to_idx = {node: idx for idx, node in enumerate(nodes)}
    # Inverse mapping to retrieve node names from indices
    idx_to_node = {idx: node for node, idx in node_to_idx.items()}
    
    # Initialize lists to store the edges in terms of indices
    data = []
    rows = []
    cols = []
    
    # Populate the edge index lists
    for edge in edge_list:
        src, dst = edge
        rows.append(node_to_idx[src])
        cols.append(node_to_idx[dst])
        data.append(1)  # Assuming unweighted graph, use 1 as the placeholder for edge existence
    
    # Number of nodes
    n_nodes = len(nodes)
    # Create the CSR matrix
    csr_graph = csr_matrix((data, (rows, cols)), shape=(n_nodes, n_nodes))
    
    # Find the connected components
    n_components, labels = connected_components(csgraph=csr_graph, directed=False, return_labels=True)
    
    # Group node indices by their component label
    subgraphs = {i: set() for i in range(n_components)}
    for node_idx, component_label in enumerate(labels):
        subgraphs[component_label].add(idx_to_node[node_idx])
    
    # Convert the dictionary to a list of sets of node names
    subgraph_node_sets = list(subgraphs.values())
    
    return subgraph_node_sets

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