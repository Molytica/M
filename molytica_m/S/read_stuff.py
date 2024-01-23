import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform

# Function to calculate pairwise distances
def calculate_distances(atom_data):
    # Using pdist and squareform to calculate pairwise distances efficiently
    return squareform(pdist(atom_data[:, 1:4]))

# Function to plot histogram
def plot_histogram(distances, protein_id):
    # Filtering distances less than 3 ångstrom
    filtered_distances = distances[distances < 3]
    plt.hist(filtered_distances, bins=100, density=True)
    plt.title(f"Distance Histogram for Protein {protein_id}")
    plt.xlabel("Distance (ångstrom)")
    plt.ylabel("Frequency")
    plt.show()

# Function to iterate over proteins and plot histograms
def plot_distance_histograms_for_proteins(species_list):
    for species in species_list:
        protein_folder = os.path.join("data/curated_chembl/opt_af_coords", species)
        for file_name in os.listdir(protein_folder):
            if file_name.endswith(".h5"):
                protein_id = file_name.split('.')[0]
                with h5py.File(os.path.join(protein_folder, file_name), 'r') as h5file:
                    atom_data = h5file['atom_data'][:]
                    distances = calculate_distances(atom_data)
                    plot_histogram(distances, protein_id)

if __name__ == "__main__":
    species = ["HUMAN", "RAT", "MOUSE", "YEAST"]
    plot_distance_histograms_for_proteins(species)
