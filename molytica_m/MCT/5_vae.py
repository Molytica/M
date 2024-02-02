from molytica_m.data_tools import alpha_fold_tools
import numpy as np
from scipy.spatial import distance
import random
import json

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
    nearest_indices = np.argsort(distances)[1:6]  # Skip the closest (itself)
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

def get_human_atom_cloud(human_uniprot_id):
    atom_cloud = np.zeros((2500, 4))
    atom_cloud[:, 0] = np.random.randint(0, 18, size=2500)
    atom_cloud[:, 1:] = np.random.randn(2500, 3)
    return atom_cloud


def get_neighbors(atom, atom_cloud):
    # Calculate Euclidean distances between the given atom and all atoms in the cloud
    distances = distance.cdist(atom_cloud[:, 1:], np.array([atom[1:]]), 'euclidean')
    # Flatten the distances array for easier indexing
    distances = distances.flatten()
    # Argsort the distances, get indices of the five closest atoms, excluding the atom itself
    nearest_indices = np.argsort(distances)[1:6]  # Skip the first index as it's the atom itself
    # Return the five closest atoms
    return atom_cloud[nearest_indices]


def get_atom_5_in_random_order():
    all_human_uniprot_ids = alpha_fold_tools.get_all_alphafold_uniprot_ids()["HUMAN"]

    for human_uniprot_id in random.sample(all_human_uniprot_ids, 20504):
        atom_cloud = get_human_atom_cloud(human_uniprot_id)
        # Replace atom types in the cloud with their embeddings
        atom_cloud_with_embeddings = replace_atom_types_with_embeddings(atom_cloud)

        for atom in random.sample(list(atom_cloud_with_embeddings), len(atom_cloud_with_embeddings)):
            neighbors = get_neighbors_with_embeddings(atom, atom_cloud_with_embeddings)
            # Now `neighbors` includes embeddings in the first 3 columns and coordinates in the next 3

            # Further processing and saving of the neighbors can be done here
            print(neighbors)


if __name__ == "__main__":
    get_atom_5_in_random_order()