import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import h5py
import random

def rotate_vectors(vectors, angle, axis):
    """ Rotate vectors by a given angle around a given axis """
    axis = axis / np.sqrt(np.dot(axis, axis))
    a = np.cos(angle / 2)
    b, c, d = -axis * np.sin(angle / 2)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    rotation_matrix = np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                                [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                                [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])
    return np.dot(vectors, rotation_matrix.T)

def get_bond_type(distance):
    # Define distance ranges for different bond types in proteins
    if distance <= 1.5:  # Typical range for a single bond
        return 'single'
    elif 1.5 < distance <= 1.7:  # Adjusted range for a double bond (less common in proteins)
        return 'double'
    # Triple bonds are extremely rare in proteins and not typically expected
    return 'none'


def find_neighbors(atom_data, atom_index, max_distance=2.6):
    atom = atom_data[atom_index, 1:4]
    neighbors = []
    for i, other_atom in enumerate(atom_data[:, 1:4]):
        if i != atom_index:
            distance = np.linalg.norm(atom - other_atom)
            if distance <= max_distance:
                neighbors.append((atom_data[i, 0], other_atom - atom))  # Store atom type and vector
    return neighbors

def kabsch_algorithm(P, Q):
    # Centroid of the points
    P_centroid = np.mean(P, axis=0)
    Q_centroid = np.mean(Q, axis=0)

    # Centre the points
    P_centered = P - P_centroid
    Q_centered = Q - Q_centroid

    # Compute the covariance matrix
    H = np.dot(P_centered.T, Q_centered)

    # Singular Value Decomposition
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)  # Rotation matrix

    # Special reflection case
    if np.linalg.det(R) < 0:
       Vt[-1,:] *= -1
       R = np.dot(Vt.T, U.T)

    return R


def plot_protein_point_cloud(species, protein_id):
    file_path = os.path.join("data/curated_chembl/opt_af_coords", species, protein_id + ".h5")
    with h5py.File(file_path, 'r') as h5file:
        atom_data = h5file['atom_data'][:]

    # Define a color map for common atom types
    atom_type_colors = {
        'C': 'black',  # Carbon
        'N': 'blue',   # Nitrogen
        'O': 'red',    # Oxygen
        'S': 'yellow', # Sulfur
        'P': 'orange', # Phosphorus
    }

    # Mapping of atom type identifiers to atom type labels
    # Update this based on how your data encodes atom types
    atom_type_labels = {
        0: 'C',
        1: 'N',
        2: 'O',
        3: 'S',
        4: 'P',
    }

    # Define colors for different bond types
    bond_colors = {
        'single': 'grey',
        'double': 'blue',
        'triple': 'orange',
        'none': 'red'  # Red to indicate no bond or unknown bond type
    }

    max_range = 0

    for atom_index in range(len(atom_data)):
        neighbors = find_neighbors(atom_data, atom_index)
        if len(neighbors) >= 3:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

            vectors = np.array([vec for _, vec in neighbors])
            centroid = np.mean(vectors, axis=0)
            centroid_normalized = centroid / np.linalg.norm(centroid)
            x_axis = np.array([1, 0, 0])
            angle = np.arccos(np.clip(np.dot(centroid_normalized, x_axis), -1.0, 1.0))
            rotation_axis = np.cross(centroid_normalized, x_axis)
            rotated_vectors = rotate_vectors(vectors, angle, rotation_axis)

            # Plot the central atom at the origin with a cross marker
            central_atom_type_label = atom_type_labels.get(atom_data[atom_index, 0], 'Unknown')
            ax.scatter(0, 0, 0, color=atom_type_colors.get(central_atom_type_label, 'gray'), marker='x', s=100, label=f'Central Atom ({central_atom_type_label})')

            # Plot rotated neighbor atoms and bonds
            max_range = 0  # Variable to keep track of the maximum range
            for (atom_type, vec), rotated_vec in zip(neighbors, rotated_vectors):
                atom_type_label = atom_type_labels.get(atom_type, 'Unknown')
                neighbor_color = atom_type_colors.get(atom_type_label, 'gray')
                ax.scatter(rotated_vec[0], rotated_vec[1], rotated_vec[2], color=neighbor_color, label=f'Atom {atom_type_label}')

                # Determine bond type based on distance and color the line accordingly
                distance = np.linalg.norm(rotated_vec)
                bond_type = get_bond_type(distance)
                bond_color = bond_colors.get(bond_type, 'red')

                # Plot the bond
                ax.plot([0, rotated_vec[0]], [0, rotated_vec[1]], [0, rotated_vec[2]], color=bond_color)

                # Add label for the bond type
                mid_point = [rotated_vec[0]/2, rotated_vec[1]/2, rotated_vec[2]/2]
                ax.text(mid_point[0], mid_point[1], mid_point[2], bond_type, color=bond_color, fontsize=8)

                # Update max_range for setting the axis limits
                max_range = max(max_range, np.max(np.abs(rotated_vec)))

            # Set the same range for each axis
            ax.set_xlim([-max_range, max_range])
            ax.set_ylim([-max_range, max_range])
            ax.set_zlim([-max_range, max_range])

            ax.set_xlabel('X Axis')
            ax.set_ylabel('Y Axis')
            ax.set_zlabel('Z Axis')
            plt.title(f"Atom {atom_index} in protein {protein_id} from species {species}")

            # Avoid duplicate labels
            handles, labels = plt.gca().get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            plt.legend(by_label.values(), by_label.keys())

            plt.show()
        else:
            print("not enough neighbors")

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
