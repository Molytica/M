from molytica_m.data_tools.create_PPI_full_combs import get_Interactome_loader
from molytica_m.ml.PPI_model import ProteinInteractionPredictor
from molytica_m.data_tools import alpha_fold_tools
import torch.optim as optim
from tqdm import tqdm
import torch.nn as nn
import numpy as np
import torch
import h5py
import sys
import os

# Check if GPU is available and use it; otherwise, use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("cuda" if torch.cuda.is_available() else "cpu")

train_loader, metadata_vector_size, graph_feature_size = get_Interactome_loader()

model = torch.load("molytica_m/ml/PPI_C_model.pth")
model.eval()

af_uniprots = alpha_fold_tools.get_alphafold_uniprot_ids()
n_af_uniprots = len(af_uniprots)

# Training loop with progress bar
num_epochs = 1  # Set the number of epochs
last_ten = []
val_max_acc = 0

save_path = 'data/PPI_interactome/'
if not os.path.exists(save_path):
    os.makedirs(save_path)

idx = 0
for epoch in range(num_epochs):
    # Initialize tqdm progress bar
    with tqdm(train_loader, unit="batch") as tepoch:
        for batch in tepoch:
            tepoch.set_description(f"Epoch {epoch+1}")

            # Load batch data
            metadata_a, metadata_b, x_a, edge_index_a, x_b, edge_index_b = batch
            metadata_a = metadata_a.to(device)
            metadata_b = metadata_b.to(device)
            x_a = x_a[0].to(device)
            edge_index_a = edge_index_a[0].to(device)
            x_b = x_b[0].to(device)
            edge_index_b = edge_index_b[0].to(device)

            # Forward and backward passes
            outputs = model(metadata_a, metadata_b, x_a, edge_index_a, x_b, edge_index_b)
            
            a_idx = idx % n_af_uniprots
            b_idx = int(idx / n_af_uniprots)

            uniprot_A = af_uniprots[a_idx]
            uniprot_B = af_uniprots[b_idx]
            value = outputs.to("cpu").detach().numpy()[0][0]

            with open(os.path.join(save_path, f"{uniprot_A}_{uniprot_B}.txt"), 'w') as file:
                file.write(str(value))
            idx += 1