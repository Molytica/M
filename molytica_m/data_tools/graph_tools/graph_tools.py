from scipy.sparse import block_diag
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from spektral.data import Graph
from rdkit.Chem import AllChem
from Bio.PDB import PDBParser
from rdkit import Chem
import numpy as np
import numpy as np
import h5py
import gzip
import os

atom_type_to_float = {
    'C': 1.0,  # Carbon
    'N': 2.0,  # Nitrogen
    'O': 3.0,  # Oxygen
    'S': 4.0,  # Sulfur
    'P': 5.0,  # Phosphorus
    'F': 6.0,  # Fluorine
    'Cl': 7.0, # Chlorine
    'Br': 8.0, # Bromine
    'I': 9.0,  # Iodine
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

def smiles_to_atom_cloud(smile): 
    # Convert the SMILES string to a molecule object
    molecule = Chem.MolFromSmiles(smile)
    if not molecule:
        raise ValueError("Invalid SMILES string")
    
    # Add hydrogens to the molecule
    molecule = Chem.AddHs(molecule)
    
    # Generate a 3D conformation for the molecule
    AllChem.EmbedMolecule(molecule, AllChem.ETKDG())
    
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
