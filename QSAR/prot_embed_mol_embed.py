import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import random
import molytica_m.chembl_curation.get_chembl_data as get_chembl_data
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.utils.data import Dataset

def load_protein_embed(uniprot_id): # 1024 dim vector
    return get_chembl_data.load_protein_embedding(uniprot_id)

def load_molecule_embedding(smiles): # 600 dim vector
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

# Example usage
dataset = get_dataset()  # Assuming this is your function to load the ChEMBL dataset
print_label_distribution(dataset)

# Seed setting for reproducibility
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
dataset = random.sample(dataset, int(float(len(dataset)) * 0.01))  # Shuffling the dataset

# Check if GPU is available and use it; otherwise, use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device in qsar:", "cuda" if torch.cuda.is_available() else "cpu")


def load_data(row):
    smiles, uniprot_id, _, _, _, _, label = row
    molecule_emb = load_molecule_embedding(smiles)
    protein_emb = load_protein_embed(uniprot_id)
    combined_emb = np.concatenate((molecule_emb, protein_emb, [0, 1]))  # binary marker for molecule and protein
    one_hot_label = get_one_hot_label(label)
    return combined_emb, one_hot_label

# Prepare the data in parallel
def prepare_data_parallel(dataset):
    X, Y = [], []
    with ProcessPoolExecutor() as executor:
        results = list(tqdm(executor.map(load_data, dataset), total=len(dataset), desc="Loading Dataset"))
    for combined_emb, one_hot_label in results:
        X.append(combined_emb)
        Y.append(one_hot_label)
    return torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.float32)

# Prepare the data
def prepare_data(dataset):
    X, Y = [], []
    for row in tqdm(dataset, desc="Loading Dataset"):
        smiles, uniprot_id, _, _, _, _, label = row
        molecule_emb = load_molecule_embedding(smiles)
        protein_emb = load_protein_embed(uniprot_id)
        combined_emb = np.concatenate((molecule_emb, protein_emb, [0, 1])) # binary marker for molecule and protein
        X.append(combined_emb)
        Y.append(get_one_hot_label(label))
    return torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.float32)

class ChEMBLDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        row = self.dataset[idx]
        smiles, uniprot_id, _, _, _, _, label = row
        molecule_emb = load_molecule_embedding(smiles)
        protein_emb = load_protein_embed(uniprot_id)
        combined_emb = np.concatenate((molecule_emb, protein_emb, [0, 1])) # binary marker for molecule and protein
        one_hot_label = get_one_hot_label(label)
        return torch.tensor(combined_emb, dtype=torch.float32), torch.tensor(one_hot_label, dtype=torch.float32)


# Assuming `dataset` is the list of data rows you previously prepared
chembl_dataset = ChEMBLDataset(dataset)

# Splitting dataset into training and validation sets
dataset_size = len(chembl_dataset)
train_size = int(dataset_size * 0.8)
val_size = dataset_size - train_size
train_dataset, val_dataset = random_split(chembl_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4)


# Define the model
class FeedForwardModel(nn.Module):
    def __init__(self):
        super(FeedForwardModel, self).__init__()
        self.fc1 = nn.Linear(1626, 512) # 1624 dims input + 1 binary marker
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 7) # 7 classes

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return torch.softmax(x, dim=1)

model = FeedForwardModel()

model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop with progress bar and SMA counters
num_epochs = 1  # Number of epochs
sma_window = 20000  # Window size for SMA calculations
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
            loss = criterion(output, target.argmax(dim=1))
            loss.backward()
            optimizer.step()

            train_loss_sma.append(loss.item())
            correct = (output.argmax(dim=1) == target.argmax(dim=1)).float().mean().item()
            train_accuracy_sma.append(correct)

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
                val_loss = criterion(output, target.argmax(dim=1))

                val_loss_sma.append(val_loss.item())
                correct = (output.argmax(dim=1) == target.argmax(dim=1)).float().mean().item()
                val_accuracy_sma.append(correct)

                if len(val_loss_sma) > sma_window:
                    val_loss_sma.pop(0)
                if len(val_accuracy_sma) > sma_window:
                    val_accuracy_sma.pop(0)

            tepoch.set_postfix(val_loss=np.mean(val_loss_sma), SMA_val_acc=np.mean(val_accuracy_sma))

    avg_val_acc = np.mean(val_accuracy_sma)
    if avg_val_acc > val_max_acc:
        print(f"New best validation accuracy: {avg_val_acc:.4f}, saving model.")
        torch.save(model, 'feed_forward_full_model.pth')  # Save the entire model
        val_max_acc = avg_val_acc