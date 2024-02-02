from molytica_m.data_tools import alpha_fold_tools
import numpy as np
from scipy.spatial import distance
import random
import json
import h5py
import os

with open("molytica_m/elements/element_embeddings_3D.json", "r") as f:
    element_embeddings = json.load(f)
    print(element_embeddings)

atom_type_to_float = {
    'C': 0.0,   # Carbon
    'N': 1.0,   # Nitrogen
    'O': 2.0,   # Oxygen
    'S': 3.0,   # Sulfur
    'P': 4.0,   # Phosphorus
    'F': 5.0,   # Fluorine
    'Cl': 6.0,  # Chlorine
    'Br': 7.0,  # Bromine
    'I': 8.0,   # Iodine
    'Na': 9.0,  # Sodium
    'K': 10.0,  # Potassium
    'B': 11.0,  # Boron
    'Si': 12.0, # Silicon
    'Se': 13.0, # Selenium
    'Li': 14.0, # Lithium
    'Zn': 15.0, # Zinc
    'Se': 17.0, # Selenium
}

def replace_atom_types_with_embeddings(atom_cloud):
    # New shape: original row count, but columns for 3 (embedding) + 3 (coordinates)
    updated_atom_cloud = np.zeros((atom_cloud.shape[0], 6))
    for i, atom in enumerate(atom_cloud):
        # Get embedding for the atom type
        embedding = get_embedding_from_int(int(atom[0]))
        if embedding is not None:
            updated_atom_cloud[i, :3] = embedding
        else:
            updated_atom_cloud[i, :3] = np.zeros(3)  # Or handle missing embedding differently
        updated_atom_cloud[i, 3:] = atom[1:]
    return updated_atom_cloud

def get_neighbors_with_embeddings(atom, atom_cloud):
    # Calculate Euclidean distances using only the coordinates (last 3 columns)
    distances = distance.cdist(atom_cloud[:, 3:], np.array([atom[3:]]), 'euclidean')
    distances = distances.flatten()
    nearest_indices = np.argsort(distances)[0:6]  # Skip the closest (itself)

    # Calculate relative positions by subtracting the atom's position from the neighbors' positions
    relative_positions = atom_cloud[nearest_indices, 3:] - atom[3:]
    atom_cloud[nearest_indices, 3:] = relative_positions

    return atom_cloud[nearest_indices]


# Invert the mapping to go from float (or int) to atom type
float_to_atom_type = {v: k for k, v in atom_type_to_float.items()}

def get_embedding_from_int(atom_int):
    # Convert integer to element symbol using the inverted mapping
    atom_type = float_to_atom_type.get(atom_int)
    if atom_type is None:
        return None  # Return None or handle error if the int doesn't correspond to any element

    # Retrieve and return the embedding from the element_embeddings dictionary
    return element_embeddings.get(atom_type)

def get_human_atom_cloud(uniprot_id, species='HUMAN'):
    # Define the path to where the atom data is saved
    file_path = os.path.join("data/curated_chembl/opt_af_coords", species, uniprot_id + ".h5")
    
    # Check if the file exists
    if not os.path.exists(file_path):
        print(f"No data found for {uniprot_id} in {species}")
        return None

    # Read the atom data from the .h5 file
    with h5py.File(file_path, 'r') as h5file:
        atom_data = h5file['atom_data'][:]
    
    return atom_data

def get_atom_5_1_in_random_order():
    all_human_uniprot_ids = alpha_fold_tools.get_all_alphafold_uniprot_ids()["HUMAN"]

    data_id = 0
    for human_uniprot_id in random.sample(all_human_uniprot_ids, 20504):
        atom_cloud = get_human_atom_cloud(human_uniprot_id)

        if atom_cloud is None:
            continue

        # Replace atom types in the cloud with their embeddings
        atom_cloud_with_embeddings = replace_atom_types_with_embeddings(atom_cloud)

        for atom in random.sample(list(atom_cloud_with_embeddings), 1):
            neighbors = get_neighbors_with_embeddings(atom, atom_cloud_with_embeddings)
            # Now `neighbors` includes embeddings in the first 3 columns and coordinates in the next 3

            print(f"Saving data for {human_uniprot_id} ({data_id})")
            save_path = f"data/curated_chembl/5_1_vae/mol/{data_id}.h5"

            if not os.path.exists(os.path.dirname(save_path)):
                os.makedirs(os.path.dirname(save_path))

            with h5py.File(save_path, 'w') as h5file:
                h5file.create_dataset('5_1_vae_data', data=neighbors)

            data_id += 1
            if data_id == 100000:
                return

            


if __name__ == "__main__":
    get_atom_5_1_in_random_order()