import os
import torch
import h5py
import sqlite3
import random
import json
from torch_geometric.data import Data
from molytica_m.data_tools.graph_tools import graph_tools_pytorch
from concurrent.futures import ProcessPoolExecutor
from molytica_m.chembl_curation import curate_chembl

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



def load_molecule_graph_form_id(mol_id, conf_id=0, target_output_path="data/curated_chembl", mol_id_to_path_preload=None):
    mol_id_to_path = {}
    if mol_id_to_path_preload is None:
        with open(os.path.join(target_output_path, "molecule_id_mappings", "mol_id_to_path.json"), 'r') as f:
            mol_id_to_path = json.load(f)
    else:
        mol_id_to_path = mol_id_to_path_preload

    file_path = mol_id_to_path[mol_id + "_" + str(conf_id)]
    
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


db_path = os.path.join("data", "curated_chembl", "SMILES_metadata.db")

def get_data_by_mol_id_or_smiles(db_path, search_value):
    # Determine if the search_value is a mol_id (integer) or smiles (string)
    if isinstance(search_value, int):
        search_column = 'mol_id'
        return_column = 'smiles'
    elif isinstance(search_value, str):
        search_column = 'smiles'
        return_column = 'mol_id'
    else:
        raise ValueError("search_value must be either an integer (mol_id) or a string (smiles)")

    # Connect to the database
    with sqlite3.connect(db_path) as conn:
        c = conn.cursor()

        # Prepare the SQL query
        query = f"SELECT {return_column}, path_comma_sep FROM molecule_index WHERE {search_column} = ?"

        # Execute the query
        c.execute(query, (search_value,))

        # Fetch the matching row
        row = c.fetchone()

        if row:
            # Extract the return value and paths
            return_value, paths = row
            path_list = paths.split(',')
            return return_value, path_list
        else:
            return None, []


def load_molecule_descriptors(smiles):
    with sqlite3.connect(db_path) as conn:
        c = conn.cursor()
        # Define the SQL command to select the row
        sql_command = "SELECT * FROM mol_metadata WHERE mol_canonical_smiles = ?"
        c.execute(sql_command, (smiles,))

        # Fetch the result
        result = c.fetchone()

    if result is None:
        descriptors = curate_chembl.calculate_descriptors(smiles)
        mol_id = curate_chembl.get_mol_id(smiles)
        with sqlite3.connect(db_path) as conn:
            c = conn.cursor()
            curate_chembl.add_mol_desc_to_db([smiles], [mol_id], [descriptors], c)
        return load_molecule_descriptors(smiles)

    # Assuming the first two columns are mol_molytica_id and mol_canonical_smiles
    descriptors = result[2:]

    return descriptors

def add_molecule_descriptors(smiles): # Takes in a list of smiles
    num_cores = os.cpu_count()
    num_workers = int(num_cores * 0.9)

    with open(os.path.join("data", "curated_chembl", "molecule_id_mappings", "smiles_to_id.json"), 'r') as f:
        smiles_to_id = json.load(f)

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        batch_descriptors = executor.map(curate_chembl.calculate_descriptors, smiles)

    mol_ids = [smiles_to_id[smile] for smile in smiles]
    batch_descriptors = list(batch_descriptors)

    with sqlite3.connect(db_path) as conn:
        c = conn.cursor()
        curate_chembl.add_mol_desc_to_db(smiles, mol_ids, batch_descriptors, c)
    
    return batch_descriptors.values()


def load_protein_metadata(uniprot_id):
    speciess = os.listdir(os.path.join("data", "curated_chembl", "af_metadata"))

    potential_file_names = []
    for species in speciess:
        potential_file_names.append(
            os.path.join("data", "curated_chembl", "af_metadata", species, f"{uniprot_id}_metadata.h5")
            )
    
    for file_name in potential_file_names:
        if not os.path.exists(file_name):
            continue

        with h5py.File(file_name, 'r') as h5file:  # Open the file in read-only mode
            metadata = h5file['metadata'][:]  # Read the entire dataset
        return metadata
    
    print(f"File not found for UniProt ID {uniprot_id}")
    return None


if __name__ == "__main__": 
    # Load the data
    #categorised_folds = get_categorised_CV_split()
    #print(categorised_folds[0][0])
    #print(len(categorised_folds[0]) * 5)

    data = load_protein_graph("P05067")
    print(data)

    seq = load_protein_sequence("P05067")
    print(seq)
    
    meta = load_protein_metadata("P05067")
    print(meta)

    aspirin_smiles = "CC(=O)OC1=CC=CC=C1C(O)=O"
    desc = load_molecule_descriptors(aspirin_smiles)
    print(desc)

    with open(os.path.join("data", "curated_chembl", "molecule_id_mappings", "smiles_to_id.json"), 'r') as f:
        smiles_to_id = json.load(f)

    smiles = list(smiles_to_id.keys())

    mol_graph_0 = load_molecule_graph(smiles[0])
    print(mol_graph_0)
    mol_graph = load_molecule_graph(aspirin_smiles)
    print(mol_graph)

    #bioactivities = get_bioactivities()
    #print(bioactivities[0])

    #folds = get_CV_split()
    #print(len(folds))