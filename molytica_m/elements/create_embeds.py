import json
import torch
from molytica_m.elements.vae import VariationalAutoencoder

# Load the trained model
model = torch.load('molytica_m/elements/vae_6D.pth')
model.eval()

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

# Extract latent space embeddings and save them
element_embeddings = {}

for key in atom_data:
    with torch.no_grad():
        data_point = torch.tensor(atom_data[key], dtype=torch.float).unsqueeze(0)
        mu, _ = model.encode(data_point)
        element_embeddings[key] = mu.squeeze().tolist()  # Convert to list for JSON serialization

# Save the embeddings as JSON
file_name = 'molytica_m/elements/element_embeddings_6D.json'
with open(file_name, 'w') as f:
    json.dump(element_embeddings, f)

print(f"Embeddings saved to {file_name}")
