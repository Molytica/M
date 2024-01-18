# Midbrinks Protein GRid Optimisation Algorithm 1
import numpy as np
from scipy.spatial.transform import Rotation as R
import random
import os
from Bio.PDB import PDBParser
import numpy as np
import gzip
import time

def get_test_points(num_points: int) -> np.ndarray:
    
    # Parameters
    std_devs = [1, 30, 100]  # Standard deviations for x, y, z

    # Generating the point cloud
    np.random.seed(0)  # For reproducibility
    point_cloud = np.random.normal(loc=0, scale=std_devs, size=(num_points, 3))



    # Define rotation in degrees
    rotation_degrees = [45, 0, 0]  # Rotation angles for x, y, z in degrees

    # Convert the rotation to a rotation matrix
    rotation_radians = np.radians(rotation_degrees)  # Convert to radians
    rotation = R.from_euler('xyz', rotation_radians)
    rotation_matrix = rotation.as_matrix()

    # Apply rotation to the point cloud
    rotated_point_cloud = np.dot(point_cloud, rotation_matrix.T)

    return rotated_point_cloud

def pdb_to_atom_cloud(pdb_file_path: str) -> np.ndarray:
    parser = PDBParser()

    # Open the gzipped PDB file
    with gzip.open(pdb_file_path, 'rt') as file:  # 'rt' mode for reading text
        structure = parser.get_structure('protein', file)

    # Extract the atom coordinates
    atom_coordinates = []
    for model in structure:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    atom_coordinates.append(atom.get_coord())

    return np.array(atom_coordinates)

def get_random_protein_cloud(species: str = "HUMAN"):
    random_protein = random.choice(os.listdir(f"data/curated_chembl/alpha_fold_data/{species}"))
    while ".pdb" not in random_protein:
        random_protein = random.choice(os.listdir(f"data/curated_chembl/alpha_fold_data/{species}"))
    
    print(random_protein.split("-")[1])
    # file is in pdb.gz format
    # Get the atom coordinates from the PDB file
    atom_coordinates = pdb_to_atom_cloud(f"data/curated_chembl/alpha_fold_data/{species}/{random_protein}")
    return atom_coordinates

def visualize_point_cloud(point_cloud):
    # Visualize the rotated atom coordinates
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Extract X, Y, and Z coordinates from point_cloud
    x = point_cloud[:, 0]
    y = point_cloud[:, 1]
    z = point_cloud[:, 2]
    
    # Scatter plot of the rotated atom coordinates
    ax.scatter(x, y, z, c='b', marker='o', label='Rotated Atoms')
    
    # Set labels for the axes
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    # Calculate the scaling factor for equal aspect ratio
    max_range = max(max(x) - min(x), max(y) - min(y), max(z) - min(z))
    mid_x = (max(x) + min(x)) / 2
    mid_y = (max(y) + min(y)) / 2
    mid_z = (max(z) + min(z)) / 2
    ax.set_xlim(mid_x - max_range / 2, mid_x + max_range / 2)
    ax.set_ylim(mid_y - max_range / 2, mid_y + max_range / 2)
    ax.set_zlim(mid_z - max_range / 2, mid_z + max_range / 2)
    
    # Set the title of the plot
    ax.set_title('Rotated Atom Coordinates')
    
    # Show the legend
    ax.legend()
    
    # Display the 3D plot
    plt.show()

def quaternion_multiply(q1, q2):
    w1, x1, y1, z1 = q1[3], q1[0], q1[1], q1[2]  # Reorder to w, x, y, z
    w2, x2, y2, z2 = q2[3], q2[0], q2[1], q2[2]  # Reorder to w, x, y, z

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    return np.array([x, y, z, w])  # Reorder back to x, y, z, w

def random_small_rotation(max_angle):
    """Generate a random rotation quaternion with an angle less than max_angle."""
    axis = np.random.normal(size=3)
    axis /= np.linalg.norm(axis)  # Normalize the axis
    angle = np.random.uniform(0, max_angle)
    rotation = R.from_rotvec(axis * angle)
    q = rotation.as_quat()
    return np.array([q[0], q[1], q[2], q[3]])  # Already in x, y, z, w format

def quaternion_angular_distance(q1, q2):
    """Calculate the angular distance between two quaternions."""
    # Normalize the quaternions
    q1 = q1 / np.linalg.norm(q1)
    q2 = q2 / np.linalg.norm(q2)

    # Reorder the quaternions to w, x, y, z for dot product calculation
    q1_reordered = np.array([q1[3], q1[0], q1[1], q1[2]])
    q2_reordered = np.array([q2[3], q2[0], q2[1], q2[2]])

    # Calculate the dot product
    dot_product = np.dot(q1_reordered, q2_reordered)
    dot_product = np.clip(dot_product, -1.0, 1.0)

    # Calculate the angular distance
    theta = 2 * np.arccos(abs(dot_product))
    return theta

def uniform_random_quaternion():
    u1, u2, u3 = np.random.rand(3)
    q1 = np.sqrt(1 - u1) * np.sin(2 * np.pi * u2)
    q2 = np.sqrt(1 - u1) * np.cos(2 * np.pi * u2)
    q3 = np.sqrt(u1) * np.sin(2 * np.pi * u3)
    q0 = np.sqrt(u1) * np.cos(2 * np.pi * u3)
    return np.array([q1, q2, q3, q0])  # x, y, z, w

def quaternion_to_rotation_matrix(quaternion):
    w, x, y, z = quaternion
    xx, xy, xz = x * x, x * y, x * z
    yy, yz, zz = y * y, y * z, z * z
    wx, wy, wz = w * x, w * y, w * z

    rotation_matrix = np.array([
        [1 - 2 * (yy + zz),     2 * (xy - wz),     2 * (xz + wy)],
        [    2 * (xy + wz), 1 - 2 * (xx + zz),     2 * (yz - wx)],
        [    2 * (xz - wy),     2 * (yz + wx), 1 - 2 * (xx + yy)]
    ])
    return rotation_matrix

def rotate_points(points, quaternion):
    rotation_matrix = quaternion_to_rotation_matrix(quaternion)
    return np.dot(points, rotation_matrix.T)

def calculate_volume(atom_coordinates: np.ndarray) -> float:
    # Use NumPy's min and max functions for efficient computation
    min_coords = np.min(atom_coordinates, axis=0)
    max_coords = np.max(atom_coordinates, axis=0)

    # Calculate the lengths of each side of the bounding box
    lengths = max_coords - min_coords

    # Calculate the volume of the bounding box
    volume = np.prod(lengths)
    return volume

def get_V(quaternion, atom_coordinates: np.ndarray) -> float:
    # Rotate the atom coordinates by the Quaternion
    
    rotated_coords = rotate_points(atom_coordinates, quaternion)
    # Calculate the volume of the rotated atom coordinates
    volume = calculate_volume(rotated_coords)

    return volume



def ProteinGridOptimiser(atom_coordinates: np.ndarray, samples_per_iteration: int = 70, iterations: int = 6) -> np.ndarray:
    
    ID_Q_V_dict = {}
    best_Vs = [calculate_volume(atom_coordinates)]

    for i in range(samples_per_iteration):
        # Generate a random Quaternion
        quaternion = uniform_random_quaternion()
        # Calculate the volume of the rotated atom coordinates
        volume = get_V(quaternion, atom_coordinates)
        # Store the Quaternion and its volume in a dictionary
        ID_Q_V_dict[i] = [quaternion, volume]
    
    # Select the 5 quaternions with least volume
    sorted_ID_Q_V_list = sorted(ID_Q_V_dict.items(), key=lambda x: x[1][1])
    # Now get the quaternions of the 5 lowest volumes
    best_quaternions = [sorted_ID_Q_V_list[i][1][0] for i in range(5)]

    """
    # Show 5 random quaternions from all the quaternions in ID_Q_V_dict
    for q_idx in random.sample(range(samples_per_iteration), 5):
        r_c = rotate_points(atom_coordinates, ID_Q_V_dict[q_idx][0])
        print(ID_Q_V_dict[q_idx][1])
        visualize_point_cloud(r_c)
    """

    """
    for q in best_quaternions:
        r_c = rotate_points(atom_coordinates, q)
        print(get_V(q, atom_coordinates))
        visualize_point_cloud(r_c)
    """

    # Print the V of the best quaternion
    best_Vs.append(sorted_ID_Q_V_list[0][1][1])

    for x in range(iterations):
        new_quaternions = {}
        id = -1
        for i, quaternion in enumerate(best_quaternions):
            # Create a random quaternion with a maximum angular distance of min_theta_row from quaternion
            rot_fact = np.random.uniform(0, 1)
            random_perturbation = random_small_rotation(rot_fact)

            for x in range(samples_per_iteration):
                id += 1
                perturbated_quaternion = quaternion_multiply(quaternion, random_perturbation)

                # Calculate the volume of the rotated atom coordinates
                volume = get_V(perturbated_quaternion, atom_coordinates)

                new_quaternions[id] = [perturbated_quaternion, volume]
        
        for x in range(5):
            # Add the best quaternions to the new quaternions dictionary
            new_quaternions[id + x + 1] = [sorted_ID_Q_V_list[x][1][0], sorted_ID_Q_V_list[x][1][1]]

        sorted_ID_Q_V_list = sorted(new_quaternions.items(), key=lambda x: x[1][1])
        best_quaternions = [sorted_ID_Q_V_list[i][1][0] for i in range(5)]
        # Print the V of the best quaternion
        best_Vs.append(sorted_ID_Q_V_list[0][1][1])
    

    # Divide best_Vs by the first element
    best_Vs = [x / best_Vs[0] for x in best_Vs]
    # return the best quaternion
    return sorted_ID_Q_V_list[0][1][0], best_Vs


def plot_volume(best_Vs):
    import matplotlib.pyplot as plt
    plt.plot(best_Vs)
    plt.show()


def get_optimised_atom_type_and_coords(atom_type_and_coordinates: np.ndarray):
    # Run the Protein Grid Optimiser
    opt_quaternion, metrics = ProteinGridOptimiser(atom_type_and_coordinates[:, 1:])

    # Rotate the atom coordinates by the optimal Quaternion
    rotated_coords = rotate_points(atom_type_and_coordinates[:, 1:], opt_quaternion)
    
    # Replace the new coordinates in the atom_type_and_coordinates array
    atom_type_and_coordinates[:, 1:] = rotated_coords

    return atom_type_and_coordinates


if __name__ == "__main__":
    start_time = time.time()
    # Load the atom coordinates
    test_coords = get_random_protein_cloud()

    print(calculate_volume(test_coords))
    # Run the Protein Grid Optimiser
    opt_quaternion, metrics = ProteinGridOptimiser(test_coords)

    # Rotate the atom coordinates by the optimal Quaternion
    rotated_coords = rotate_points(test_coords, opt_quaternion)

    # Visualize the rotated atom coordinates
    print("--- %s seconds ---" % (time.time() - start_time))
    plot_volume(metrics)
    visualize_point_cloud(rotated_coords)
    
