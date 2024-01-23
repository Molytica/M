import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
import os
import h5py
import numpy as np

def plot_protein_point_cloud(species, protein_id):
    file_path = os.path.join("data/curated_chembl/opt_af_coords", species, protein_id + ".h5")
    with h5py.File(file_path, 'r') as h5file:
        atom_data = h5file['atom_data'][:]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot using atom coordinates
    ax.scatter(atom_data[:, 1], atom_data[:, 2], atom_data[:, 3], c=atom_data[:, 0], cmap='viridis')

    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')
    plt.title(f"Point cloud of protein {protein_id} from species {species}")

    plt.show()

def select_random_protein(species):
    species_folder = os.path.join("data/curated_chembl/opt_af_coords", species)
    protein_files = [f for f in os.listdir(species_folder) if f.endswith('.h5')]
    if not protein_files:
        raise ValueError(f"No protein files found for species {species}")
    random_protein_file = random.choice(protein_files)
    protein_id = random_protein_file.split('.')[0]
    return protein_id

if __name__ == "__main__":
    species = "HUMAN"  # You can choose from ["HUMAN", "RAT", "MOUSE", "YEAST"]
    protein_id = select_random_protein(species)
    plot_protein_point_cloud(species, protein_id)
