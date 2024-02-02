from molytica_m.data_tools import alpha_fold_tools
import numpy as np
from scipy.spatial import distance

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


def get_human_atom_cloud(human_uniprot_id):

    atom_cloud = np.zeros((2500, 4))

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


if __name__ == "__main__":
    all_human_uniprot_ids = alpha_fold_tools.get_all_alphafold_uniprot_ids()["HUMAN"]

    for human_uniprot_id in all_human_uniprot_ids:
        atom_cloud = get_human_atom_cloud(human_uniprot_id)

        for atom in atom_cloud:
            neightbors = get_neighbors(atom, atom_cloud)

            print(neightbors)


        # Save the five closest neighbors.