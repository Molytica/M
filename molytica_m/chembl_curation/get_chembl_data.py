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

def add_molecule_data(db_path, mol_id, smiles, path_list):
    # Validate input types
    if not isinstance(mol_id, int):
        raise ValueError("mol_id must be an integer")
    if not isinstance(smiles, str):
        raise ValueError("smiles must be a string")
    if not isinstance(path_list, list) or not all(isinstance(p, str) for p in path_list):
        raise ValueError("path_list must be a list of strings")

    # Convert path_list to a comma-separated string
    paths_comma_sep = ','.join(path_list)

    try:
        # Connect to the database
        with sqlite3.connect(db_path) as conn:
            c = conn.cursor()

            # Prepare the SQL query
            query = "INSERT INTO molecule_index (mol_id, smiles, path_comma_sep) VALUES (?, ?, ?)"

            # Execute the query
            c.execute(query, (mol_id, smiles, paths_comma_sep))

            # Commit the transaction
            conn.commit()

            print("Molecule data added successfully.")

            # Close the cursor
            c.close()

    except sqlite3.IntegrityError:
        print("Error: mol_id already exists in the database.")
    except sqlite3.Error as e:
        raise sqlite3.Error(f"Database error: {e}")
    except Exception as e:
        raise Exception(f"An error occurred: {e}")

def load_molecule_graph(search_key, conf_id=0, db_path="data/curated_chembl/molecule_index.db"):
    # Determine the type of search_key (mol_id or smiles)
    search_column = 'mol_id' if isinstance(search_key, int) else 'smiles'

    # Connect to the database and fetch the path list
    try:
        with sqlite3.connect(db_path) as conn:
            c = conn.cursor()
            query = f"SELECT path_comma_sep FROM molecule_index WHERE {search_column} = ?"
            c.execute(query, (search_key,))
            row = c.fetchone()

            if not row:
                print(f"No data found for {search_column}: {search_key}")
                return None

            path_list = row[0].split(',') if row[0] else []
            if conf_id >= len(path_list):
                print(f"Invalid conf_id {conf_id} for {search_column}: {search_key}")
                return None

            file_path = path_list[conf_id]

    except sqlite3.Error as e:
        print(f"Database error: {e}")
        return None

    # Check if the file exists
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return None

    # Load the graph pytorch geometric data
    pyg_data = curate_chembl.load_mol_pyg_data_from_hdf5(file_path)

    return pyg_data

def add_molecule_to_library(smiles, db_path="data/curated_chembl/molecule_index.db"):
    # Validate input types
    if not isinstance(smiles, str):
        raise ValueError("smiles must be a string")

    # Calculate the descriptors
    descriptors = curate_chembl.calculate_descriptors(smiles)

    # Get the maximum mol_id from the database
    try:
        with sqlite3.connect(db_path) as conn:
            c = conn.cursor()
            c.execute("SELECT MAX(mol_id) FROM molecule_index")
            max_mol_id = c.fetchone()[0]
            if max_mol_id is None:
                max_mol_id = 0
            new_mol_id = max_mol_id + 1

            # Check if the new mol_id already exists in the database
            c.execute("SELECT COUNT(*) FROM molecule_index WHERE mol_id = ?", (new_mol_id,))
            count = c.fetchone()[0]
            if count > 0:
                raise ValueError(f"mol_id {new_mol_id} already exists in the database")

            

            # Add the molecule to the database
            curate_chembl.add_mol_desc_to_db([smiles], [new_mol_id], [descriptors], c)
    except sqlite3.Error as e:
        print(f"Database error: {e}")
        return None

    return new_mol_id

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

def get_data_by_mol_id_or_smiles(search_value, db_path="data/curated_chembl/molecule_index.db"):
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

    #data = load_protein_graph("P05067")
    #print(data)

    #seq = load_protein_sequence("P05067")
    #print(seq)
    
    #meta = load_protein_metadata("P05067")
    #print(meta)

    aspirin_smiles = "CC(=O)OC1=CC=CC=C1C(O)=O"
    #desc = load_molecule_descriptors(aspirin_smiles)
    #print(desc)

    with open(os.path.join("data", "curated_chembl", "molecule_id_mappings", "smiles_to_id.json"), 'r') as f:
        smiles_to_id = json.load(f)

    smiles = list(smiles_to_id.keys())

    get_data_by_mol_id_or_smiles("COc1ccc(CN[C@@H](C(=O)N[C@H](C(=O)NCc2ccc(OC)cc2O)C(C)C)[C@H](O)[C@H](Cc2ccccc2)NC(=O)[C@@H](NC(=O)Cc2cccc3ccccc23)C(C)(C)C)cc1")

    smiles, path = get_data_by_mol_id_or_smiles(2)
    mol_graph_0 = load_molecule_graph(smiles)
    print(mol_graph_0)
    mol_graph = load_molecule_graph(2)
    print(mol_graph)

    #bioactivities = get_bioactivities()
    #print(bioactivities[0])

    #folds = get_CV_split()
    #print(len(folds))