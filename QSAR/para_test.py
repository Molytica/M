import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold
import multiprocessing
import numpy as np
import random
import molytica_m.chembl_curation.get_chembl_data as get_chembl_data
from molytica_m.arch2 import chemBERT
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.utils.data import Dataset
from rdkit import Chem
from rdkit.Chem import inchi
import h5py
import sys
import os


def load_protein_embed(uniprot_id): # 1024 dim vector
    return get_chembl_data.load_protein_embedding(uniprot_id)

def load_molecule_embedding(smiles): # 600 dim vector'
    inchi_key = smiles_to_inchikey(smiles)
    if inchi_key is not None:
        for folder in os.listdir("data/curated_chembl/mole_embeds"):
            if os.path.exists(f"data/curated_chembl/mole_embeds/{folder}/{inchi_key}.h5"):
                with h5py.File(f"data/curated_chembl/mole_embeds/{folder}/{inchi_key}.h5", "r") as f:
                    return f[inchi_key][:]
    return get_chembl_data.load_molecule_embedding(smiles)

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
from transformers import AutoTokenizer, AutoModelForMaskedLM
dataset = random.sample(dataset, int(float(len(dataset)) * 1))  # Shuffling the dataset

def save_mol_embeds_to_h5py_parallel_helper(args):
    print("Starting helper")
    smiles, file_name, skip_if_exists = args
    print(f"Smiles {smiles}")
    print(f"File name {file_name}")
    print(f"Skip if exists {skip_if_exists}")
    if os.path.exists(file_name) and skip_if_exists:
        return

    tok, mod = chemBERT.get_chemBERT_tok_mod()
    molecule_emb = chemBERT.get_molecule_mean_logits(smiles, tok, mod)[0][0]
    print(molecule_emb.shape)

    print("Saving...")
    with h5py.File(file_name, "w") as f:
        f.create_dataset("mol_mean_logits", data=molecule_emb)
    print("Helper Finished")


"""def process(input_smiles):
    print("In process")
    tok, mod = get_chemBERT_tok_mod()
    shape_1 = get_molecule_mean_logits(input_smiles, tok, mod)
    print(shape_1[0][0].shape)
    print("Done with process")"""


def save_mol_embeds_to_h5py_parallel(dataset):
    folder_count = 0
    folder_row_count = 0

    args = []
    batch_size = 2

    for row in tqdm(dataset, "Preparing args"):
        folder_row_count += 1
        smiles = row[0]
        inchi_key = smiles_to_inchikey(smiles)

        if inchi_key is None:
            continue
        
        file_name = f"data/curated_chembl/mol_embeds/{folder_count}/{inchi_key}.h5"
        if not os.path.exists(f"data/curated_chembl/mol_embeds/{folder_count}"):
                os.makedirs(f"data/curated_chembl/mol_embeds/{folder_count}")


        if folder_row_count % 100000 == 0:
            folder_count += 1
            folder_row_count = 0
        
        if not os.path.exists(file_name) or True:
            args.append((smiles, file_name, False))
            if len(args) > batch_size: # Batch size 1000
                print(args)
                with multiprocessing.Pool(processes=2) as pool:
                    list(tqdm(pool.imap(save_mol_embeds_to_h5py_parallel_helper, args), total=len(args)))
                args = []

    return list(args)


def get_args(dataset, fraq = 1.0):
    folder_count = 0
    folder_row_count = 0

    args = []

    for row in tqdm(dataset, "Preparing args"):
        folder_row_count += 1
        smiles = row[0]
        inchi_key = smiles_to_inchikey(smiles)

        if inchi_key is None:
            continue
        
        file_name = f"data/curated_chembl/mol_embeds/{folder_count}/{inchi_key}.h5"
        if not os.path.exists(f"data/curated_chembl/mol_embeds/{folder_count}"):
                os.makedirs(f"data/curated_chembl/mol_embeds/{folder_count}")

        if folder_row_count % 100000 == 0:
            folder_count += 1
            folder_row_count = 0
        
        args.append((smiles, file_name, False))

        if len(args) > int(len(dataset) * fraq):
            break

    return list(args)

dataset = get_dataset()
#save_mol_embeds_to_h5py_parallel(dataset)
print(smiles_to_inchikey(dataset[0][0]))
print(smiles_to_inchikey(dataset[1][0]))
sys.exit(0)
