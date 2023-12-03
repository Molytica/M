from molytica_m.data_tools.graph_tools import graph_tools
from molytica_m.data_tools import alpha_fold_tools
from concurrent.futures import ProcessPoolExecutor
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data
from collections import Counter
from Bio.PDB import PDBParser
from tqdm import tqdm
import numpy as np
import itertools
import random
import torch
import json
import h5py
import gzip
import sys
import os


af_uniprots = alpha_fold_tools.get_alphafold_uniprot_ids()
n_af_uniprots = len(af_uniprots)

with open("molytica_m/data_tools/filtered_no_reverse_duplicates_huri_and_biogrid_uniprot_edges.json", "r") as file:
    edge_list = json.load(file)["filtered_no_reverse_duplicates_huri_and_biogrid_uniprot_edges"]

random.shuffle(edge_list)

n_edges = len(edge_list)
split = []
train_frac = 0.8
val_frac = 0.1

for idx, edge in enumerate(edge_list):
    if idx / n_edges < train_frac:
        split.append("train")
    elif idx / n_edges < train_frac + val_frac:
        split.append("val")
    else:
        split.append("test")

"""
print(n_edges)
print(len(split))

occurrences = Counter(split)

for occ in occurrences.values():
    print(occ / n_edges)

"""


# Full PPI


def get_metadata(af_uniprot):
    file_name = f"data/af_metadata/{af_uniprot}_metadata.h5"
    
    with h5py.File(file_name, 'r') as h5file:
        # Read the dataset 'metadata'
        metadata_vector = h5file['metadata'][:]
    
    return metadata_vector


def extract_af_protein_graph(arg_tuple):
    input_folder_path, output_folder_path, af_uniprot_id = arg_tuple
    input_file_name = os.path.join(input_folder_path, f"AF-{af_uniprot_id}-F1-model_v4.pdb.gz")
    output_file_name = os.path.join(output_folder_path, f"{af_uniprot_id}_graph.h5")
    
    with gzip.open(input_file_name, 'rt') as file:
        parser = PDBParser(QUIET=True)
        atom_structure = parser.get_structure("", file)

    protein_atom_cloud_array = []
    for model in atom_structure:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    atom_type_numeric = graph_tools.get_atom_type_numeric(atom.element.strip())
                    atom_data = [atom_type_numeric, *atom.get_coord()]
                    protein_atom_cloud_array.append(atom_data)
    
    protein_atom_cloud_array = np.array(protein_atom_cloud_array)
    atom_point_cloud_atom_types = protein_atom_cloud_array[:, 0]  # Changed from :1 to 0 for correct indexing
    n_atom_types = 9

    # One-hot encode the atom types
    features = np.eye(n_atom_types)[atom_point_cloud_atom_types.astype(int) - 1]

    # Create the graph
    graph = graph_tools.csr_graph_from_point_cloud(protein_atom_cloud_array[:, 1:], STANDARD_BOND_LENGTH=1.5)

    # Convert CSR graph to PyTorch Geometric format
    edge_index = np.vstack(graph.nonzero())
    edge_attr = graph.data

    # Save the graph in h5 file
    with h5py.File(output_file_name, 'w') as h5file:
        h5file.create_dataset('edge_index', data=edge_index)
        h5file.create_dataset('edge_attr', data=edge_attr)
        h5file.create_dataset('atom_features', data=features)  # Save atom types as features
    

def create_af_atom_clouds():
    input_folder_path = "data/alpha_fold_data/"
    output_folder_path = "data/af_protein_1_dot_5_angstrom_graphs/"

    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    arg_tuples = []
    for af_uniprot_id in af_uniprots:
        arg_tuples.append((input_folder_path, output_folder_path, af_uniprot_id))

    with ProcessPoolExecutor() as executor:
        results = list(tqdm(executor.map(extract_af_protein_graph, arg_tuples), desc="Creating protein atom clouds", total=len(arg_tuples)))
    

create_af_atom_clouds()

class ProteinInteractionDataset(Dataset):
    def __init__(self):
        pass

    def __len__(self):
        return 20504**2

    def __getitem__(self, idx):

        prot_A_uniprot_idx = int(idx / n_af_uniprots)
        prot_B_uniprot_idx = idx % n_af_uniprots
        uniprot_A = af_uniprots[prot_A_uniprot_idx]
        uniprot_B = af_uniprots[prot_B_uniprot_idx]

        metadata_a = get_metadata(uniprot_A)
        metadata_b = get_metadata(uniprot_B)

        graph_data_a = get_graph_data(uniprot_A)
        graph_data_b = get_graph_data(uniprot_B)

        if (uniprot_A, uniprot_B) in edge_list or (uniprot_B, uniprot_A) in edge_list:
            label = 1
        else:
            label = 0

        return metadata_a, metadata_b, graph_data_a, graph_data_b, label

        

# PPI pos + equal number PPI neg (half positive half negative)