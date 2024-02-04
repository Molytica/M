import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold
from concurrent.futures import ProcessPoolExecutor
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


def get_full_prot_embed(uniprot_id): # 1024 dim vector
    embed_path = "data/curated_chembl/padded_protein_full_embeddings/HUMAN/" + uniprot_id + "_embedding.h5"
    
    with h5py.File(embed_path, "r") as f:
        return f['embedding'][:]

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

def get_mol_embed(smiles):
    inchi_key = smiles_to_inchikey(smiles)
    if inchi_key is not None:
        for folder in os.listdir("data/curated_chembl/mol_embeds"):
            if os.path.exists(f"data/curated_chembl/mol_embeds/{folder}/{inchi_key}.h5"):
                with h5py.File(f"data/curated_chembl/mol_embeds/{folder}/{inchi_key}.h5", "r") as f:
                    embed = f["mol_mean_embed"][:]
                    if len(embed) != 600:
                        print(f"Embedding for {smiles} is of length {len(embed)}")
                    return embed
    print(f"Could not find smiles {smiles} in mol_embeds")
    return None

def get_full_mol_embed(smiles):
    inchi_key = smiles_to_inchikey(smiles)
    if inchi_key is not None:
        for folder in os.listdir("data/curated_chembl/full_mol_embeds"):
            if os.path.exists(f"data/curated_chembl/full_mol_embeds/{folder}/{inchi_key}.h5"):
                with h5py.File(f"data/curated_chembl/full_mol_embeds/{folder}/{inchi_key}.h5", "r") as f:
                    return f["full_mol_embed"][:]
    return None

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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device in qsar:", "cuda" if torch.cuda.is_available() else "cpu")

dataset = get_dataset()
"""
save_mol_embeds_to_h5py(dataset)
sys.exit(0)
"""

class ChEMBLDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        row = self.dataset[idx]
        smiles, uniprot_id, _, _, _, _, label = row
        molecule_emb = get_mol_embed(smiles)
        protein_emb = get_full_prot_embed(uniprot_id)
        
        protein_emb_flat = protein_emb.flatten()

        combined_emb = np.concatenate((molecule_emb, protein_emb_flat))
        one_hot_label = get_one_hot_label(label)
        return torch.tensor(combined_emb, dtype=torch.float32), torch.tensor(one_hot_label, dtype=torch.float32)


# Assuming `dataset` is the list of data rows you previously prepared
chembl_dataset = ChEMBLDataset(dataset)

# Splitting dataset into training and validation sets
dataset_size = len(chembl_dataset)
train_size = int(dataset_size * 0.8)
val_size = dataset_size - train_size
train_dataset, val_dataset = random_split(chembl_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=16)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=16)

def calculate_class_weights(dataset):
    label_counts = {}
    for _, _, _, _, _, _, label in dataset:
        label_counts[label] = label_counts.get(label, 0) + 1

    total_samples = len(dataset)
    weights = [total_samples / label_counts[label] for label in sorted(label_counts)]
    return torch.tensor(weights, dtype=torch.float32)

class_weights = calculate_class_weights(dataset)
class_weights = class_weights.to(device)
print("Class Weights:", class_weights)

class CustomTransformerLayer(nn.Module):
    def __init__(self, dim_model, dim_feedforward, dropout_rate=0.1):
        super(CustomTransformerLayer, self).__init__()
        # Define multi-head attention mechanisms with different numbers of heads
        self.attention1 = nn.MultiheadAttention(dim_model, 4)
        
        self.norm1 = nn.LayerNorm(dim_model)
        self.norm2 = nn.LayerNorm(dim_model)
        
        # Define feed-forward networks for each attention mechanism
        self.feed_forward1 = nn.Sequential(
            nn.Linear(dim_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, dim_model)
        )
        
        # Add 50% dropout layer
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, src):
        src2 = self.norm1(src)
        attn_output, _ = self.attention1(src2, src2, src2)
        src = src + attn_output
        src = self.norm2(src)
        
        # Apply 50% dropout after the attention mechanism
        src = self.dropout(src)
        
        src = src + self.feed_forward1(src)
        
        return src

class TransformerModel(nn.Module):
    def __init__(self, input_dim, dim_feedforward, num_classes, num_layers=5, dropout_rate=0.5):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(input_dim, dim_feedforward)
        
        # Optionally, add dropout after embedding
        self.embedding_dropout = nn.Dropout(dropout_rate)
        
        self.layers = nn.ModuleList([
            CustomTransformerLayer(dim_feedforward, dim_feedforward, dropout_rate=dropout_rate)
            for _ in range(num_layers)
        ])
        
        # Define the feedforward layers with dropout after ReLU
        self.feed_forward_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim_feedforward, dim_feedforward),
                nn.ReLU(),
                nn.Dropout(dropout_rate),  # Add dropout here
            )
            for _ in range(3)  # Number of feedforward layers
        ])
        
        # Output layer
        self.output_layer = nn.Linear(dim_feedforward, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = self.embedding_dropout(x)  # Apply dropout after embedding
        
        for layer in self.layers:
            x = layer(x)
        
        # Apply the feedforward layers
        for feed_forward_layer in self.feed_forward_layers:
            x = feed_forward_layer(x)
        
        # Apply the output layer
        x = self.output_layer(x)
        
        return x  # Return raw logits


input_dim = 1024 * 512 + 600  # Assuming 1024x512 for protein, 600 for molecule

# Update the dimensions of the feedforward layers
dim_feedforward = 512  # Adjust this based on your requirements
num_classes = 7  # Number of classes for the output layer
num_layers = 5  # Number of transformer layers

# Initialize the model
print("Creating model")
model = TransformerModel(input_dim=input_dim, 
                         dim_feedforward=dim_feedforward, 
                         num_classes=num_classes, 
                         num_layers=num_layers)

#print("Loading model")
#model = torch.load('molytica_m/QSAR/TransQSAR2.pth')
model.to(device)

# Define loss function and optimizer
# Use BCEWithLogitsLoss for one-hot encoded labels
criterion = nn.CrossEntropyLoss()
# Trial 1
# Lr = 0.01
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop with progress bar and SMA counters
num_epochs = 300  # Number of epochs
sma_window = 20000 // 64  # Window size for SMA calculations
val_max_acc = 0
for epoch in range(num_epochs):
    model.train()
    train_loss_sma, train_accuracy_sma = [], []
    with tqdm(train_loader, unit="batch") as tepoch:
        for data, target in tepoch:
            tepoch.set_description(f"Epoch {epoch+1}")

            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)

            # Convert one-hot encoded targets to class indices for CrossEntropyLoss
            _, target_classes = torch.max(target, 1)

            # Compute loss
            loss = criterion(output, target_classes)  # target_classes are class indices
            loss.backward()
            optimizer.step()

            # Get the predicted class as the index of the maximum value in each row
            _, predicted_classes = torch.max(output, 1)

            # Calculate accuracy as the mean of correct predictions
            correct = (predicted_classes == target_classes).float().mean()
            train_accuracy_sma.append(correct.item())

            # Append the loss
            train_loss_sma.append(loss.item())

            if len(train_loss_sma) > sma_window:
                train_loss_sma.pop(0)
            if len(train_accuracy_sma) > sma_window:
                train_accuracy_sma.pop(0)

            tepoch.set_postfix(loss=np.mean(train_loss_sma), SMA_acc=np.mean(train_accuracy_sma))

    val_loss_sma, val_accuracy_sma = [], []
    model.eval()
    with tqdm(val_loader, unit="batch") as tepoch:
        for data, target in tepoch:
            tepoch.set_description(f"Validating Epoch {epoch+1}")

            data, target = data.to(device), target.to(device)

            with torch.no_grad():
                output = model(data)
                # Convert one-hot encoded targets to class indices for CrossEntropyLoss
                _, target_classes = torch.max(target, 1)

                # Compute loss
                loss = criterion(output, target_classes)  # target_classes are class indices

                # Get the predicted class as the index of the maximum value in each row
                _, predicted_classes = torch.max(output, 1)

                # Calculate accuracy as the mean of correct predictions
                correct = (predicted_classes == target_classes).float().mean()
                val_accuracy_sma.append(correct.item())

                # Append the loss
                val_loss_sma.append(loss.item())

                # Manage SMA window
                if len(val_loss_sma) > sma_window:
                    val_loss_sma.pop(0)
                if len(val_accuracy_sma) > sma_window:
                    val_accuracy_sma.pop(0)

            tepoch.set_postfix(val_loss=np.mean(val_loss_sma), SMA_val_acc=np.mean(val_accuracy_sma))

    avg_val_acc = np.mean(val_accuracy_sma)
    if avg_val_acc > val_max_acc:
        print(f"New best validation accuracy: {avg_val_acc:.4f}, saving model.")
        torch.save(model, 'molytica_m/QSAR/TransQSAR2.pth')  # Save the entire model
        val_max_acc = avg_val_acc