import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold
import multiprocessing
import numpy as np
import random
from molytica_m.arch2 import chemBERT
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.utils.data import Dataset
from rdkit import Chem
from rdkit.Chem import inchi
import h5py
import sys
import os

def save_mol_embeds_to_h5py_parallel(args):
    smiles, file_name, skip_if_exists, models, locks = args

    for i in range(len(models)):
        with locks[i]:  # This automatically acquires and releases the lock
            # Get the model
            tok, mod = models[i]

            # Generate embedding
            print("Generating embedding")
            molecule_emb = chemBERT.get_molecule_mean_logits(smiles, tok, mod)[0][0]
            print("Embedding generated")
            # Generate folder if it does not exist and save embedding to file
            folder = "/".join(file_name.split("/")[:-1])
            if not os.path.exists(folder):
                os.makedirs(folder)
            with h5py.File(file_name, "w") as f:
                f.create_dataset("mol_mean_logits", data=molecule_emb)
            break

def convert_SMILES_to_embed_h5():
    models = [chemBERT.get_chemBERT_tok_mod() for _ in range(10)]
    print("Models created")
    
    dataset = chemBERT.get_dataset()
    print("Dataset obtained")
    
    args = chemBERT.get_args(dataset)
    print("Arguments prepared")

    with multiprocessing.Manager() as manager:
        locks = [manager.Lock() for _ in range(10)]
        print("Locks created")

        wrapped_args = [(*arg, models, locks) for arg in args]
        print("Arguments wrapped")

        with multiprocessing.Pool(10) as p:
            print("Parallel processing started")
            
            list(tqdm(p.map(save_mol_embeds_to_h5py_parallel, wrapped_args), total=len(wrapped_args)))
            
            print("Parallel processing completed")


if __name__ == "__main__":
    convert_SMILES_to_embed_h5()
