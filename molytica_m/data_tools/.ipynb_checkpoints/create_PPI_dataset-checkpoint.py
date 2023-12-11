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

with open("molytica_m/data_tools/filtered_no_reverse_duplicates_huri_and_biogrid_af_uniprot_edges.json", "r") as file:
    edge_list = json.load(file)["filtered_no_reverse_duplicates_huri_and_biogrid_af_uniprot_edges"]

random.shuffle(edge_list)

n_edges = len(edge_list)
edges_split = {"train": [], "val": [], "test": []}
train_frac = 0.8
val_frac = 0.1

for idx, edge in enumerate(edge_list):
    if idx / n_edges < train_frac:
        edges_split["train"].append(edge)
    elif idx / n_edges < train_frac + val_frac:
        edges_split["val"].append(edge)
    else:
        edges_split["test"].append(edge)

"""
print(n_edges)
print(len(split))

occurrences = Counter(split)

for occ in occurrences.values():
    print(occ / n_edges)

"""

def get_metadata(af_uniprot):
    file_name = f"data/af_metadata/{af_uniprot}_metadata.h5"
    
    with h5py.File(file_name, 'r') as h5file:
        # Read the dataset 'metadata'
        metadata_vector = h5file['metadata'][:]
    
    return metadata_vector

def get_graph(af_uniprot):
    file_name = f"data/af_protein_1_dot_5_angstrom_graphs/{af_uniprot}_graph.h5"

    with h5py.File(file_name, 'r') as h5file:
        edge_index = torch.tensor(h5file['edge_index'][:], dtype=torch.long)
        edge_attr = torch.tensor(h5file['edge_attr'][:], dtype=torch.float)
        atom_features = torch.tensor(h5file['atom_features'][:], dtype=torch.float)

    return atom_features, edge_index

def get_graph_raw(af_uniprot):
    file_name = f"data/af_protein_1_dot_5_angstrom_graphs/{af_uniprot}_graph.h5"

    with h5py.File(file_name, 'r') as h5file:
        edge_index = h5file['edge_index'][:]
        edge_attr = h5file['edge_attr'][:]
        atom_features = h5file['atom_features'][:]

    return atom_features, edge_index

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

class ProteinInteractionDataset(Dataset):
    def __init__(self, edges):
        self.edges = edges
        self.length = len(edges) * 2 # times two because half are positive and half are negative

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        
        if idx % 2 == 0:
            idx_half = int(idx / 2)
            uniprot_A = self.edges[idx_half][0]
            uniprot_B = self.edges[idx_half][1]
            label = 1
        else:
            uniprot_A = random.choice(af_uniprots)
            uniprot_B = random.choice(af_uniprots)

            while [uniprot_A, uniprot_B] in edge_list or [uniprot_B, uniprot_A] in edge_list:
                uniprot_A = random.choice(af_uniprots)
                uniprot_B = random.choice(af_uniprots)
            label = 0

        metadata_a = get_metadata(uniprot_A)
        metadata_b = get_metadata(uniprot_B)

        x_a, edge_index_a = get_graph(uniprot_A)
        x_b, edge_index_b = get_graph(uniprot_B)

        return (
            torch.tensor(metadata_a, dtype=torch.float),
            torch.tensor(metadata_b, dtype=torch.float),
            torch.tensor(x_a, dtype=torch.float),
            torch.tensor(edge_index_a, dtype=torch.long),
            torch.tensor(x_b, dtype=torch.float),
            torch.tensor(edge_index_b, dtype=torch.long),
            torch.tensor([label], dtype=torch.float)
        )



def get_data_loader_and_size():
    train_data_loader = DataLoader(ProteinInteractionDataset(edges_split["train"]), batch_size=1, shuffle=True, num_workers=10)
    val_data_loader = DataLoader(ProteinInteractionDataset(edges_split["val"]), batch_size=1, shuffle=True, num_workers=10)
    test_data_loader = DataLoader(ProteinInteractionDataset(edges_split["test"]), batch_size=1, shuffle=True, num_workers=10)

    metadata_vector_size = len(get_metadata("A0A0A0MRZ7"))
    graph_feature_size = 9

    return train_data_loader, val_data_loader, test_data_loader, metadata_vector_size, graph_feature_size


if __name__ == "__main__":
    create_af_atom_clouds()