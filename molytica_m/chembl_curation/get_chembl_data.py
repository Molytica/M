import os
import torch
import h5py
import sqlite3
import random
import json
from torch_geometric.data import Data
from molytica_m.data_tools.graph_tools import graph_tools_pytorch

def load_protein_graph(uniprot_id, fold_n=1):
    speciess = os.listdir(os.path.join("data", "curated_chembl", "af_protein_1_dot_5_angstrom_graphs"))

    potential_file_names = []
    for species in speciess:
        potential_file_names.append(
            os.path.join("data", "curated_chembl", "af_protein_1_dot_5_angstrom_graphs", species, f"{uniprot_id}_F{fold_n}_graph.h5")
            )
    
    for file_name in potential_file_names:
        if not os.path.exists(file_name):
            continue
        # Load the graph
        with h5py.File(file_name, 'r') as h5file:
            edge_index = h5file['edge_index'][()]
            edge_attr = h5file['edge_attr'][()]
            atom_features = h5file['atom_features'][()]
        
        #convert to pytorch data
        edge_index = torch.tensor(edge_index, dtype=torch.long)
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)
        atom_features = torch.tensor(atom_features, dtype=torch.float)

        data = Data(x=atom_features, edge_index=edge_index, edge_attr=edge_attr)

        return data
    print(f"File not found for UniProt ID {uniprot_id}")
    return None

def load_protein_sequence(uniprot_id):
    speciess = os.listdir(os.path.join("data", "curated_chembl", "protein_sequences"))

    potential_file_names = []
    for species in speciess:
        potential_file_names.append(
            os.path.join("data", "curated_chembl", "protein_sequences", species, f"{uniprot_id}_sequence.h5")
            )
    
    for file_name in potential_file_names:
        if not os.path.exists(file_name):
            continue
        # Load and decode the sequence
        with h5py.File(file_name, 'r') as h5file:
            encoded_sequence = h5file['sequence'][()]
            decoded_sequence = encoded_sequence.decode('utf-8')
        
        return decoded_sequence
    
    print(f"File not found for UniProt ID {uniprot_id}")
    return None


def load_molecule_graph(mol_id, conf_id=0, target_output_path="data/curated_chembl"):
    mol_id_to_path = {}
    with open(os.path.join(target_output_path, "molecule_id_mappings", "mol_id_to_path.json"), 'r') as f:
        mol_id_to_path = json.load(f)

    file_path = mol_id_to_path[mol_id + "_" + conf_id]
    
    # Check if the file exists
    if not os.path.exists(file_path):
        print(f"File not found for molecule ID {mol_id}")
        return None

    # Load the graph
    features, csr = graph_tools_pytorch.load_features_csr_matrix_from_hdf5(file_path)
    
    return features, csr

def get_bioactivities():
    # Connect to the SQLite database
    conn = sqlite3.connect(os.path.join("data", "curated_chembl", "smiles_alphafold_v4_human_uniprot_chembl_bioactivities.db"))
    cursor = conn.cursor()

    # Execute the SQL query to fetch all rows from the bioactivities table
    cursor.execute("SELECT * FROM bioactivities")
    rows = cursor.fetchall()

    # Close the database connection
    conn.close()

    # Return the rows as an array
    return rows

    
def get_CV_split():
    rows = get_bioactivities()
    random.seed(42)
    random.shuffle(rows)

    # Split into 5 folds
    fold_size = len(rows) // 5
    folds = []
    for i in range(5):
        folds.append(rows[i*fold_size:(i+1)*fold_size])

    return folds



def categorize_inhibitor(ic50_nm):
    """
    Categorize an inhibitor based on its IC50 value in nanomolar (nM), and assign a revised numerical value.
    In this revised version, 3 represents a strong inhibitor and 0 represents weak or no significant inhibition.

    Parameters:
    ic50_nm (float): The IC50 value in nM.

    Returns:
    tuple: A tuple containing the category of the inhibitor (string) and its revised numerical value (int).
    """
    if ic50_nm == None:
        return "No Data", None

    ic50_nm = float(ic50_nm)
    if ic50_nm < 10:
        return "Strong Inhibitor", 3
    elif 10 <= ic50_nm < 100:
        return "Medium Inhibitor", 2
    elif 100 <= ic50_nm < 1000:
        return "Light Inhibitor", 1
    else:
        return "Weak or No Significant Inhibition", 0


def get_categorised_CV_split():
    rows = get_bioactivities()
    modified_rows = []

    for row in rows:
        effect_type = None
        num_cat = None
        if row[2] == "IC50":
            effect_type = -1
            cat, num_cat = categorize_inhibitor(row[4])
        elif row[2] == "EC50":
            effect_type = 1
            cat, num_cat = categorize_inhibitor(row[4]) # Assume this applies for EC50 as well
        elif row[2] == "Ki":
            effect_type = -1
            cat, num_cat = categorize_inhibitor(row[4]) # Assume this applies for EC50 as well
        
        if effect_type is not None and num_cat is not None:
            modulation_effect = num_cat * effect_type
            new_row = list(row) + [modulation_effect]
            modified_rows.append(new_row)  
        

    random.seed(42)
    random.shuffle(modified_rows)

    # Split into 5 folds
    fold_size = len(modified_rows) // 5
    folds = []
    for i in range(5):
        folds.append(modified_rows[i*fold_size:(i+1)*fold_size])

    return folds

if __name__ == "__main__": 
    # Load the data
    categorised_folds = get_categorised_CV_split()
    print(categorised_folds[0][0])
    print(len(categorised_folds[0]) * 5)

    data = load_protein_graph("P05067")
    print(data)

    seq = load_protein_sequence("P05067")
    print(seq)

    bioactivities = get_bioactivities()
    print(bioactivities[0])

    folds = get_CV_split()
    print(len(folds))