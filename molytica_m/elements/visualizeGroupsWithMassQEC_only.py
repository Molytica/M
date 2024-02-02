import json
import torch
import plotly.graph_objects as go
from molytica_m.elements.vae import VariationalAutoencoder

# Load the trained model
model = torch.load('molytica_m/elements/vae_qec_only_1k.pth')
model.eval()

# Load JSON data
with open("molytica_m/elements/element_vectors.json", "r") as f:
    atom_data = json.load(f)

# Extract atomic masses and store in a dictionary
atomic_masses = {key: atom_data[key][2] for key in atom_data}

with open("molytica_m/elements/element_vectors_qec_only.json", "r") as f:
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

# Extract latent space embeddings
latent_space_embeddings = []

for key in atom_data:
    with torch.no_grad():
        data_point = torch.tensor(atom_data[key], dtype=torch.float).unsqueeze(0)
        mu, _ = model.encode(data_point)
        latent_space_embeddings.append((mu.squeeze().numpy(), key))

# Group categorization (simplified for main groups)
element_groups = {
    'Alkali Metals': ['Li', 'Na', 'K', 'Rb', 'Cs', 'Fr'],
    'Alkaline Earth Metals': ['Be', 'Mg', 'Ca', 'Sr', 'Ba', 'Ra'],
    'Transition Metals': ['Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
                          'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd',
                          'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg'],
    'Metalloids': ['B', 'Si', 'Ge', 'As', 'Sb', 'Te'],
    'Nonmetals': ['H', 'C', 'N', 'O', 'P', 'S', 'Se'],
    'Halogens': ['F', 'Cl', 'Br', 'I', 'At'],
    'Noble Gases': ['He', 'Ne', 'Ar', 'Kr', 'Xe', 'Rn']
}

# Function to find the group of an element
def find_group(element):
    for group, elements in element_groups.items():
        if element in elements:
            return group
    return "Other"

# Colors for each group
group_colors = {
    'Alkali Metals': 'blue',
    'Alkaline Earth Metals': 'green',
    'Transition Metals': 'grey',
    'Metalloids': 'yellow',
    'Nonmetals': 'orange',
    'Halogens': 'purple',
    'Noble Gases': 'red',
    'Other': 'black'
}

# Prepare data for the Plotly 3D scatter plot
scatter_data = []
size_factor = 0.4
text_font = dict(family="Arial, sans-serif", size=12, color="black")  # Example text styling

for embedding, label in latent_space_embeddings:
    group = find_group(label)
    color = group_colors[group]
    size = (atomic_masses[label] * size_factor) ** (1/3)
    scatter_data.append(go.Scatter3d(
        x=[embedding[0]],
        y=[embedding[1]],
        z=[embedding[2]],
        mode='markers+text',
        marker=dict(size=size, color=color),
        text=label,
        textposition="bottom center",
        textfont=text_font
    ))

# Create the 3D scatter plot
fig = go.Figure(data=scatter_data)
fig.update_layout(
    scene=dict(
        xaxis_title='Latent Dim 1',
        yaxis_title='Latent Dim 2',
        zaxis_title='Latent Dim 3'
    )
)

fig.show()