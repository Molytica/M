import torch
import numpy as np
import h5py
import os
import random
from torch.utils.data import Dataset
from molytica_m.MCT._5_1_vae_prot_t import VAE

class H5Dataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.file_names = os.listdir(root_dir)

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        file_path = os.path.join(self.root_dir, self.file_names[idx])
        with h5py.File(file_path, 'r') as file:
            data = file['5_1_vae_data'][:]
        # Normalize data to be between 0 and 1
        data_min = data.min(axis=0, keepdims=True)
        data_max = data.max(axis=0, keepdims=True)
        normalized_data = (data - data_min) / (data_max - data_min)
        normalized_data = np.nan_to_num(normalized_data)  # Handle divisions by zero
        return torch.tensor(normalized_data, dtype=torch.float), self.file_names[idx]

def load_model(model_path):
    model = torch.load(model_path)
    model.eval()  # Set the model to evaluation mode
    return model

def reconstruct_and_compare(model, dataset):
    idx = random.randint(0, len(dataset) - 1)
    protein_data, file_name = dataset[idx]
    print(f"Original data for {file_name}:")
    print(protein_data)

    # Assuming the model expects a batch dimension, we unsqueeze at 0 to add it
    protein_data_batch = protein_data.unsqueeze(0)
    with torch.no_grad():  # Inference only, no gradients
        reconstructed, _, _ = model(protein_data_batch)
    reconstructed_data = reconstructed.squeeze(0)  # Remove the batch dimension
    print(f"Reconstructed data for {file_name}:")
    print(reconstructed_data)

if __name__ == "__main__":
    # Path to your trained model and data directory
    model_path = 'molytica_m/MCT/5_1_vae.pth'
    data_dir = 'data/curated_chembl/5_1_vae/prot'

    # Load the trained VAE model
    vae_model = load_model(model_path)
    
    # Initialize the dataset
    dataset = H5Dataset(root_dir=data_dir)

    # Reconstruct and compare the protein data
    reconstruct_and_compare(vae_model, dataset)
