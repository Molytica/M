import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
import os
import h5py
import numpy as np
from molytica_m.data_tools import alpha_fold_tools
import sys

def plot_protein_point_cloud(species, protein_id):
    file_path = os.path.join("data/curated_chembl/opt_af_coords", species, protein_id + ".h5")
    with h5py.File(file_path, 'r') as h5file:
        atom_data = h5file['atom_data'][:]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot using atom coordinates
    ax.scatter(atom_data[:, 1], atom_data[:, 2], atom_data[:, 3], c=atom_data[:, 0], cmap='viridis')

    # Find global max and min for x, y, z coordinates
    overall_min = np.min(atom_data[:, 1:4])
    overall_max = np.max(atom_data[:, 1:4])

    # Set axis limits to the same range for x, y, z
    ax.set_xlim(overall_min, overall_max)
    ax.set_ylim(overall_min, overall_max)
    ax.set_zlim(overall_min, overall_max)

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


def get_sequence_lengths():
    dims = []
    for species in ["HUMAN"]:
        for uniprot in alpha_fold_tools.get_all_alphafold_uniprot_ids()[species]:
            file_path = os.path.join("data/curated_chembl/protein_full_embeddings", species, uniprot + "_embedding.h5")
            with h5py.File(file_path, 'r') as h5file:
                embedding = h5file['embedding'][:]
            print(embedding.shape[0])
            dims.append(embedding.shape[0])
    
    return dims

if __name__ == "__main__":
    max_length_get = max(get_sequence_lengths())
    sys.exit(0)
    species = "HUMAN"  # You can choose from ["HUMAN", "RAT", "MOUSE", "YEAST"]
    protein_id = select_random_protein(species)
    plot_protein_point_cloud(species, protein_id)
