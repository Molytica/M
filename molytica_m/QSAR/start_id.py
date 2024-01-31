import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import random
from molytica_m.arch2 import chemBERT
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.utils.data import Dataset
from rdkit import Chem
from rdkit.Chem import inchi
from molytica_m.chembl_curation import get_chembl_data
import h5py
import sys
import os

def load_molecule_embedding(smiles): # 600 dim vector
    return get_chembl_data.gen_full_molecule_embedding(smiles)

def get_dataset():
    return get_chembl_data.get_categorised_data()

def get_one_hot_label(label):
    value_list = [-3 , -2 , -1 , 0 , 1 , 2 , 3]
    one_hot_label = [0] * 7
    one_hot_label[value_list.index(label)] = 1
    return one_hot_label

def print_label_distribution(dataset):
    label_counts = {}
    total_samples = len(dataset)

    for _, _, _, _, _, _, label in dataset:
        if label not in label_counts:
            label_counts[label] = 0
        label_counts[label] += 1

    print("Label Distribution in the Dataset:")
    for label, count in label_counts.items():
        proportion = (count / total_samples) * 100
        print(f"Label {label}: {count} occurrences, {proportion:.2f}% of the dataset")

def smiles_to_inchikey(smiles):
    """
    Convert a SMILES string to an InChIKey.

    Parameters:
    smiles (str): A SMILES string representing a chemical compound.

    Returns:
    str: The corresponding InChIKey.
    """
    try:
        # Convert SMILES to RDKit Molecule
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError("Invalid SMILES string.")

        # Generate InChIKey
        inchi_key = inchi.MolToInchiKey(mol)
        return inchi_key
    except Exception as e:
        return str(e)


# Example usage
dataset = get_dataset()  # Assuming this is your function to load the ChEMBL dataset
print_label_distribution(dataset)

# Seed setting for reproducibility
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
dataset = random.sample(dataset, int(float(len(dataset)) * 1))  # Shuffling the dataset

# Check if GPU is available and use it; otherwise, use CPU
device = torch.device("cpu")
print(f"Device in qsar is {device}")

def save_mol_embeds_to_h5py(dataset, start_id=0):
    folder_count = 0
    folder_row_count = 0
    total_count = 0

    for row in tqdm(dataset, "converting to h5"):
        total_count += 1
        folder_row_count += 1
        smiles = row[0]
        inchi_key = smiles_to_inchikey(smiles)

        if inchi_key is None:
            continue

        if not os.path.exists(f"data/curated_chembl/full_mol_embeds/{folder_count}/{inchi_key}.h5") and total_count >= start_id:
            molecule_emb = load_molecule_embedding(smiles)
            if not os.path.exists(f"data/curated_chembl/full_mol_embeds/{folder_count}"):
                os.makedirs(f"data/curated_chembl/full_mol_embeds/{folder_count}")

            with h5py.File(f"data/curated_chembl/full_mol_embeds/{folder_count}/{inchi_key}.h5", "w") as f:
                f.create_dataset("full_mol_embed", data=molecule_emb)

        if folder_row_count % 100000 == 0:
            folder_count += 1
            folder_row_count = 0


dataset = get_dataset()
# Import start id as argument
start_id = int(sys.argv[1])
print(f"Start id is {start_id}")
save_mol_embeds_to_h5py(dataset, start_id=start_id)
sys.exit(0)