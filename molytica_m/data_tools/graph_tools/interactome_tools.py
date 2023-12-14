from molytica_m.data_tools.alpha_fold_tools import get_alphafold_uniprot_ids
from itertools import combinations_with_replacement
import networkx as nx
from tqdm import tqdm
import pandas as pd
import os, json
import random

def get_neighbors_from_uniprots(edge_list, uniprot_ids, n_step_neighbors=3):
    # Create a graph from the edge list
    G = nx.Graph()
    G.add_edges_from(edge_list)
    
    # Find all neighbors for each UniProt ID up to n_neighbors away
    uniprot_edges_of_n_neighbors = []
    for uniprot_id in uniprot_ids:
        if uniprot_id in G:
            # Get all nodes within n_neighbors hops from uniprot_id
            neighbors = nx.single_source_shortest_path_length(G, uniprot_id, cutoff=n_step_neighbors)
            # Get edges between uniprot_id and its neighbors
            for neighbor, distance in neighbors.items():
                if distance > 0:  # Exclude self-loops
                    uniprot_edges_of_n_neighbors.append((uniprot_id, neighbor))
    
    return uniprot_edges_of_n_neighbors

def in_HuRI(uniprot_pair_list, edge_list):
    if uniprot_pair_list in edge_list or [uniprot_pair_list[1], uniprot_pair_list[0]] in edge_list:
        print("in HuRI")
        return True
    else:
        print("not in HuRI")
        return False

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

def get_full_edge_list():
    with open("molytica_m/data_tools/filtered_no_reverse_duplicates_huri_and_biogrid_af_uniprot_edges.json", "r") as file:
        return json.load(file)["filtered_no_reverse_duplicates_huri_and_biogrid_af_uniprot_edges"]
