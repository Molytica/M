import json
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

def label_to_onehot(label):
    onehot = torch.zeros(len(all_labels))
    onehot[label_to_index[label]] = 1
    return onehot

class AtomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        atom_vector, label_onehot = self.data[idx]
        return atom_vector, label_onehot
    
# Define the VAE
class VariationalAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VariationalAutoencoder, self).__init__()

        # Encoder
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, latent_dim)  # Mean
        self.fc22 = nn.Linear(hidden_dim, latent_dim)  # Variance

        # Decoder
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, input_dim)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        h1 = self.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = self.relu(self.fc3(z))
        return self.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

def vae_loss_function(recon_x, x, mu, logvar):
    BCE = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

if __name__ == "__main__":
    # Load JSON data
    with open("molytica_m/elements/element_vectors.json", "r") as f:
        atom_data = json.load(f)

    # Calculate means for each feature, excluding None
    feature_sums = [0] * len(next(iter(atom_data.values())))
    counts = [0] * len(feature_sums)

    for key in atom_data:
        for i, value in enumerate(atom_data[key]):
            if value is not None:
                feature_sums[i] += value
                counts[i] += 1

    means = [feature_sum / count if count > 0 else 0 for feature_sum, count in zip(feature_sums, counts)]

    # Replace None with mean values
    for key in atom_data:
        atom_data[key] = [value if value is not None else means[i] for i, value in enumerate(atom_data[key])]

    # Find max and min values for each feature
    max_values = [max(filter(lambda v: v is not None, (atom_data[key][i] for key in atom_data))) for i in range(len(feature_sums))]
    min_values = [min(filter(lambda v: v is not None, (atom_data[key][i] for key in atom_data))) for i in range(len(feature_sums))]

    # Scale data to range [0, 1]
    for key in atom_data:
        atom_data[key] = [(value - min_values[i]) / (max_values[i] - min_values[i]) if value is not None else (means[i] - min_values[i]) / (max_values[i] - min_values[i]) for i, value in enumerate(atom_data[key])]

    # Remaining code is unchanged
    # Create a list of all unique labels
    all_labels = sorted(set(atom_data.keys()))
    label_to_index = {label: idx for idx, label in enumerate(all_labels)}


    # Convert data to a list of tuples (atom_vector, label)
    data = [(torch.tensor(atom_data[key], dtype=torch.float), label_to_onehot(key)) for key in atom_data]

    # Split data into training and testing sets
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)


    train_dataset = AtomDataset(train_data)
    test_dataset = AtomDataset(test_data)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


    # Model parameters
    input_dim = len(train_data[0][0])  # Size of atom vector
    hidden_dim = 50
    latent_dim = 3

    model = VariationalAutoencoder(input_dim, hidden_dim, latent_dim)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Training loop
    num_epochs = 100000
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for data, _ in train_loader:
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data)
            loss = vae_loss_function(recon_batch, data, mu, logvar)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
        print(f'Epoch {epoch}, Loss: {train_loss / len(train_loader.dataset)}')

    # Evaluate the model
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for data, _ in test_loader:
            recon, mu, logvar = model(data)
            test_loss += vae_loss_function(recon, data, mu, logvar).item()
    print(f'Test loss: {test_loss / len(test_loader.dataset)}')

    # save the full model
    torch.save(model, 'molytica_m/elements/vae42.pth')

    # Example of using the model for inference
    with torch.no_grad():
        sample = test_data[0][0].unsqueeze(0)  # Take the first data point
        reconstructed, _, _ = model(sample)
        print('Original:', sample)
        print('Reconstructed:', reconstructed)
