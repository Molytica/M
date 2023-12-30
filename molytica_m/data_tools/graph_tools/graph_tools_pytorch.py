from molytica_m.data_tools.graph_tools import interactome_tools
from scipy.sparse.csgraph import connected_components
from molytica_m.data_tools import alpha_fold_tools
from scipy.sparse import block_diag
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from rdkit.Chem import AllChem
from Bio.PDB import PDBParser
from rdkit import Chem
from tqdm import tqdm
import networkx as nx
import numpy as np
import random
import h5py
import gzip
import os

atom_type_to_float = {
    'C': 0.0,  # Carbon
    'N': 1.0,  # Nitrogen
    'O': 2.0,  # Oxygen
    'S': 3.0,  # Sulfur
    'P': 4.0,  # Phosphorus
    'F': 5.0,  # Fluorine
    'Cl': 6.0, # Chlorine
    'Br': 7.0, # Bromine
    'I': 8.0,  # Iodine
    'Na': 9.0, # Sodium
    'K': 10.0, # Potassium
    'B': 11.0, # Boron
    'Si': 12.0,# Silicon
    'Se': 13.0,# Selenium
    'Li': 14.0,# Lithium
    'Zn': 15.0,# Zinc
    'As': 16.0,# Arsenic
    'Se': 17.0,# Selenium
}

def csr_graph_from_point_cloud(atom_point_cloud, STANDARD_BOND_LENGTH=1.5):
    # Build a k-d tree for quick nearest-neighbor lookup
    tree = cKDTree(atom_point_cloud[:,1:])
    
    # Query the tree for pairs within the bond length
    pairs = tree.query_pairs(r=STANDARD_BOND_LENGTH)
    
    # Create row index and column index arrays for CSR format
    row_idx = np.array([pair[0] for pair in pairs])
    col_idx = np.array([pair[1] for pair in pairs])
    
    # Create data array for CSR format (all ones, assuming single bond)
    data = np.ones(len(pairs))
    
    # Create the CSR matrix
    csr_graph = csr_matrix((data, (row_idx, col_idx)), shape=(len(atom_point_cloud), len(atom_point_cloud)))
    
    return csr_graph

def get_atom_type_numeric(atom_type):
    return atom_type_to_float.get(atom_type, 0)

def smiles_to_atom_cloud(smile, minimize_energy=True): 
    # Convert the SMILES string to a molecule object
    molecule = Chem.MolFromSmiles(smile)
    if not molecule:
        raise ValueError("Invalid SMILES string")
    
    # Add hydrogens to the molecule
    molecule = Chem.AddHs(molecule)
    
    # Generate a 3D conformation for the molecule
    AllChem.EmbedMolecule(molecule, AllChem.ETKDG())
    
    if minimize_energy:
        # Minimize the energy of the conformation
        AllChem.UFFOptimizeMolecule(molecule)
    
    # Extract the atom types and 3D coordinates of the atoms
    conf = molecule.GetConformer()
    atom_cloud_data = []
    for idx, atom in enumerate(molecule.GetAtoms()):
        if atom.GetSymbol() != 'H':
            atom_type = atom_type_to_float[atom.GetSymbol()]
            position = conf.GetAtomPosition(idx)
            atom_cloud_data.append((atom_type, position.x, position.y, position.z))

    # Convert the atom cloud data to a NumPy array
    atom_cloud_array = np.array(atom_cloud_data, dtype=float)
    return atom_cloud_array

def smiles_to_atom_multiple_clouds(smile, num_conformations=5, minimize_energy=True):
    # Convert the SMILES string to a molecule object
    molecule = Chem.MolFromSmiles(smile)
    if not molecule:
        raise ValueError("Invalid SMILES string")
    
    # Add hydrogens to the molecule
    molecule = Chem.AddHs(molecule)
    
    # Generate multiple 3D conformations for the molecule
    ids = AllChem.EmbedMultipleConfs(molecule, numConfs=num_conformations, params=AllChem.ETKDG())

    atom_clouds = []
    for conf_id in ids:
        if minimize_energy:
            # Minimize the energy of the conformation
            AllChem.UFFOptimizeMolecule(molecule, confId=conf_id)

        # Extract the atom types and 3D coordinates of the atoms
        conf = molecule.GetConformer(conf_id)
        atom_cloud_data = []
        for idx, atom in enumerate(molecule.GetAtoms()):
            if atom.GetSymbol() != 'H':
                atom_type = atom_type_to_float[atom.GetSymbol()]
                position = conf.GetAtomPosition(idx)
                atom_cloud_data.append((atom_type, position.x, position.y, position.z))

        # Convert the atom cloud data to a NumPy array and add to list
        atom_cloud_array = np.array(atom_cloud_data, dtype=float)
        atom_clouds.append(atom_cloud_array)

    return atom_clouds

def combine_graphs(graphs):
    # Check if the graphs list is empty
    if not graphs:
        raise ValueError("The list of graphs is empty")

    # Initialize lists to hold combined adjacency matrices and features
    combined_adjacencies = []
    combined_features = []

    # Iterate over each graph and collect their adjacency matrices and features
    for graph in graphs:
        combined_adjacencies.append(graph.a)
        combined_features.append(graph.x)

    # Combine the adjacency matrices using block diagonal form
    combined_adjacency = block_diag(combined_adjacencies)

    # Combine the feature matrices vertically
    combined_features = np.vstack(combined_features)

    # Create a Spektral Graph object with the combined data
    combined_graph = Graph(x=combined_features, a=combined_adjacency)

    return combined_graph

def get_graph_from_smiles_string(smiles_string):
    atom_cloud = smiles_to_atom_cloud(smiles_string)
    csr_matrix = csr_graph_from_point_cloud(atom_cloud)

    if np.max(atom_cloud[:, 0]) > 9:
        print(f"UNKNOWN ATOM TYPE ============================================================={np.max(atom_cloud[:, 0])}")
    atom_point_cloud_atom_types = atom_cloud[:, 0] # get the atom types
    n_atom_types = 9

    features = np.eye(n_atom_types)[atom_point_cloud_atom_types.astype(int) - 1] 

    return Graph(x = features, a=csr_matrix)

def get_raw_graph_from_smiles_string(smiles_string):
    atom_cloud = smiles_to_atom_cloud(smiles_string, minimize_energy=True)
    csr_matrix = csr_graph_from_point_cloud(atom_cloud)

    if np.max(atom_cloud[:, 0]) > 9:
        print(f"UNKNOWN ATOM TYPE ============================================================={np.max(atom_cloud[:, 0])}")
    atom_point_cloud_atom_types = atom_cloud[:, 0] # get the atom types
    n_atom_types = 9

    features = np.eye(n_atom_types)[atom_point_cloud_atom_types.astype(int) - 1] 

    return features, csr_matrix

def get_raw_graphs_from_smiles_string(smiles_string, num_conformations=5, minimize_energy=True):
    atom_clouds = smiles_to_atom_multiple_clouds(smiles_string, num_conformations=num_conformations, minimize_energy=minimize_energy)

    feature_list = []
    csr_matrix_list = []

    for atom_cloud in atom_clouds:
        atom_point_cloud_atom_types = atom_cloud[:, 0]  # Get the atom types
        n_atom_types = len(atom_type_to_float.keys())

        features = np.eye(n_atom_types)[atom_point_cloud_atom_types.astype(int)]
        csr_matrix = csr_graph_from_point_cloud(atom_cloud)

        feature_list.append(features)
        csr_matrix_list.append(csr_matrix)

    return feature_list, csr_matrix_list

def save_features_csr_matrix_to_hdf5(features, csr_matrix, file_path):
    with h5py.File(file_path, 'w') as f:
        # Save features
        f.create_dataset('features', data=features)

        # Save CSR matrix components
        f.create_dataset('data', data=csr_matrix.data)
        f.create_dataset('indices', data=csr_matrix.indices)
        f.create_dataset('indptr', data=csr_matrix.indptr)
        f.create_dataset('shape', data=csr_matrix.shape)

def get_graph_from_uniprot_id(uniprot_id):
    folder_path = "data/alpha_fold_data/"
    file_name = os.path.join(folder_path, f"AF-{uniprot_id}-F1-model_v4.pdb.gz")
    
    with gzip.open(file_name, 'rt') as file:
        parser = PDBParser(QUIET=True)
        atom_structure = parser.get_structure("", file)

    protein_atom_cloud_array = []
    for model in atom_structure:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    atom_type_numeric = get_atom_type_numeric(atom.element.strip())
                    atom_data = [atom_type_numeric, *atom.get_coord()]
                    protein_atom_cloud_array.append(atom_data)
    
    protein_atom_cloud_array = np.array(protein_atom_cloud_array)
    atom_point_cloud_atom_types = protein_atom_cloud_array[:, 0]  # Changed from :1 to 0 for correct indexing
    n_atom_types = 9

    # One-hot encode the atom types
    features = np.eye(n_atom_types)[atom_point_cloud_atom_types.astype(int) - 1]  # Make sure the types are integers

    # Now features is a two-dimensional numpy array with one-hot encoding
    csr_graph = csr_graph_from_point_cloud(protein_atom_cloud_array)

    return Graph(x=features, a=csr_graph)

def save_graph(G, file_path):
    if not os.path.exists(os.path.dirname(file_path)):
        os.makedirs(os.path.dirname(file_path))

    csr_adjacency = G.a.tocsr() if not isinstance(G.a, csr_matrix) else G.a

    with h5py.File(file_path, 'w') as f:
        # Save the features matrix
        f.create_dataset('features', data=G.x)
        # Save the adjacency matrix in CSR format
        f.create_dataset('data', data=csr_adjacency.data)
        f.create_dataset('indices', data=csr_adjacency.indices)
        f.create_dataset('indptr', data=csr_adjacency.indptr)
        f.create_dataset('shape', data=csr_adjacency.shape)
        # Save the labels or targets if they exist
        if hasattr(G, 'y') and G.y is not None:
            f.create_dataset('labels', data=G.y)

def load_graph(file_path):
    with h5py.File(file_path, 'r') as f:
        # Load the features matrix
        features = f['features'][:]

        # Load the CSR components and reconstruct the adjacency matrix
        data = f['data'][:]
        indices = f['indices'][:]
        indptr = f['indptr'][:]
        shape = f['shape'][:]
        adjacency = csr_matrix((data, indices, indptr), shape=shape)

        # Load the labels or targets if they exist
        label = f['labels'][:] if 'labels' in f else None

    # Create the Spektral Graph object
    graph = Graph(x=features, a=adjacency, y=label)

    return graph

def get_neighbors_from_uniprots(edge_list, uniprot_ids, n_step_neighbors=3):
    # Create a graph from the edge list
    G = nx.Graph()
    G.add_edges_from(edge_list)
    
    # Find all neighbors for each UniProt ID up to n_neighbors away
    uniprot_edges_of_n_neighbors = []
    for uniprot_id in uniprot_ids:
        if uniprot_id in G:
            # Get all nodes within n_neighbors hops from uniprot_id
            neighbors = nx.single_source_shortest_path_length(G, uniprot_id, cutoff=n_step_neighbors)
            # Get edges between uniprot_id and its neighbors
            for neighbor, distance in neighbors.items():
                if distance > 0:  # Exclude self-loops
                    uniprot_edges_of_n_neighbors.append((uniprot_id, neighbor))
    
    return uniprot_edges_of_n_neighbors

def get_only_unique(edge_list):
    unique_pairs = list(set(tuple(sorted(pair)) for pair in edge_list))
    unique_list = [list(pair) for pair in unique_pairs]
    return unique_list

def both_in_uniprot_list(edge, uniprot_list): # AlphaFold
    if edge[0] in uniprot_list and edge[1] in uniprot_list:
        return True
    return False

def extract_unique_triplets(data):
    unique_triplets = []
    for triplet in data:
        if triplet not in unique_triplets:
            unique_triplets.append(triplet)
    return unique_triplets

def get_edges_from_tree(tree_n: list, interesting_uniprot_ids: list):
    interactome_edge_list = interactome_tools.get_HuRI_table_as_uniprot_edge_list()
    af_uniprots = alpha_fold_tools.get_alphafold_uniprot_ids()
    edges_to_evaluate = []

    step = 0
    for n in tree_n:
        step += 1
        step_edges = get_neighbors_from_uniprots(interactome_edge_list, interesting_uniprot_ids, n_step_neighbors=step)
        print(step_edges)
        step_edges = get_only_unique(step_edges) # Might already be unique but double check
        step_edges = [[edge[0], edge[1], step] for edge in step_edges if both_in_uniprot_list(edge, af_uniprots)] # Filter only AF edges (because of structure data limitation)
        print(step_edges)
        random.shuffle(step_edges)

        if n is not None and n < len(step_edges): # If n, sample, else take all edges
            step_edges = random.sample(step_edges, n)
        
        edges_to_evaluate += step_edges

    print(edges_to_evaluate)
    unique_triplets = extract_unique_triplets(edges_to_evaluate)
    return unique_triplets

def get_nodes_from_tree(tree_n: list, interesting_uniprot_ids: list):
    interactome_edge_list = interactome_tools.get_HuRI_table_as_uniprot_edge_list()
    af_uniprots = alpha_fold_tools.get_alphafold_uniprot_ids()
    nodes_to_evaluate = set()

    step = 0
    for n in tree_n:
        step += 1
        step_edges = get_neighbors_from_uniprots(interactome_edge_list, interesting_uniprot_ids, n_step_neighbors=step)
        step_edges = get_only_unique(step_edges)  # Ensure edges are unique
        step_edges = [[edge[0], edge[1], step] for edge in step_edges if both_in_uniprot_list(edge, af_uniprots)]  # Filter only AF edges
        # Extract unique nodes from the step_edges
        unique_nodes = set()

        for item in step_edges:
            unique_nodes.add((item[1], item[2]))
        print(unique_nodes)
        # Randomly sample n nodes if n is not None, else take all nodes
        if n is not None and n < len(unique_nodes):
            sampled_nodes = set(random.sample(unique_nodes, n))
        else:
            sampled_nodes = unique_nodes

        nodes_to_evaluate.update(sampled_nodes)

    return list(nodes_to_evaluate)


def get_subgraph_nodes_from_edgelist(edge_list):
    # Create a set of all nodes
    nodes = set(node for edge in edge_list for node in edge)
    # Create a mapping from node names to integers
    node_to_idx = {node: idx for idx, node in enumerate(nodes)}
    # Inverse mapping to retrieve node names from indices
    idx_to_node = {idx: node for node, idx in node_to_idx.items()}
    
    # Initialize lists to store the edges in terms of indices
    data = []
    rows = []
    cols = []
    
    # Populate the edge index lists
    for edge in edge_list:
        src, dst = edge
        rows.append(node_to_idx[src])
        cols.append(node_to_idx[dst])
        data.append(1)  # Assuming unweighted graph, use 1 as the placeholder for edge existence
    
    # Number of nodes
    n_nodes = len(nodes)
    # Create the CSR matrix
    csr_graph = csr_matrix((data, (rows, cols)), shape=(n_nodes, n_nodes))
    
    # Find the connected components
    n_components, labels = connected_components(csgraph=csr_graph, directed=False, return_labels=True)
    
    # Group node indices by their component label
    subgraphs = {i: set() for i in range(n_components)}
    for node_idx, component_label in enumerate(labels):
        subgraphs[component_label].add(idx_to_node[node_idx])
    
    # Convert the dictionary to a list of sets of node names
    subgraph_node_sets = list(subgraphs.values())
    
    return subgraph_node_sets

def save_graphs(PPI_molecules, save_path, split_key):
    skipped = 0
    idx = -1
    for key in tqdm(sorted(list(PPI_molecules.keys())), desc="Generating and saving data", unit="c_graphs"):
        idx += 1
        try:
            iPPI = PPI_molecules[key]
            file_name = os.path.join(save_path, split_key, f'{idx}_{split_key}.h5')

            G = combine_graphs([get_graph_from_uniprot_id(iPPI['proteins'][0]),
                                        get_graph_from_uniprot_id(iPPI['proteins'][0]),
                                        get_graph_from_smiles_string(iPPI['molecule'])])
            G.y = np.array([iPPI["iPPI"]])
            save_graph(G, file_name)
        except:
            skipped += 1


def main():
    print("main")

if __name__ == "__main__":
    main()