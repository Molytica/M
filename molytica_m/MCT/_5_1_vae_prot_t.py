import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
import os
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim, dropout_rate=0.5):
        super(VAE, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),  # Add dropout after activation
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),  # Add dropout after activation
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),  # Add dropout after activation
            nn.Linear(128, latent_dim * 2)  # *2 for mean and log variance
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),  # Add dropout after activation
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),  # Add dropout after activation
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),  # Add dropout after activation
            nn.Linear(512, input_dim),
            nn.Sigmoid()  # Assuming input is normalized between 0 and 1
        )
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, x):
        # Encode
        encoded = self.encoder(x)
        mu, log_var = encoded.chunk(2, dim=-1)  # Split encoded values into mu and log_var
        
        # Reparameterize
        z = self.reparameterize(mu, log_var)
        
        # Decode
        return self.decoder(z), mu, log_var

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
        return torch.tensor(normalized_data, dtype=torch.float)

def train_vae(dataset, epochs=50, batch_size=64, learning_rate=1e-3):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = VAE(input_dim=dataset[0].shape[1], latent_dim=8)  # Assuming input_dim from dataset
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_function = nn.BCELoss(reduction='sum')  # Binary Cross-Entropy Loss

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        with tqdm(total=len(dataloader), desc=f'Epoch {epoch+1}') as pbar:
            for batch in dataloader:
                optimizer.zero_grad()
                reconstructed, mu, log_var = model(batch)
                # Calculate loss
                bce = loss_function(reconstructed, batch)
                kl_divergence = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
                loss = bce + kl_divergence
                loss.backward()
                train_loss += loss.item()
                optimizer.step()
                pbar.update(1)
                pbar.set_postfix({'Loss': train_loss / len(dataloader.dataset)})
        
        # Save the model after each epoch
        model_save_path = f'molytica_m/MCT/5_1_vae.pth'
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        torch.save(model, model_save_path)  # Save the full model
        print(f'Model saved to {model_save_path}. Epoch {epoch+1}, Loss: {train_loss / len(dataloader.dataset)}')

        print(f'Epoch {epoch+1}, Loss: {train_loss / len(dataloader.dataset)}')

if __name__ == "__main__":
    # Initialize and train your model
    dataset = H5Dataset(root_dir='data/curated_chembl/5_1_vae/prot')
    train_vae(dataset)

