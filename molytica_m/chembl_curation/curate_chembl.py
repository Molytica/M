import os, sqlite3
from concurrent.futures import ProcessPoolExecutor
from molytica_m.data_tools import id_mapping_tools
from molytica_m.data_tools.graph_tools import graph_tools_pytorch
from molytica_m.data_tools import alpha_fold_tools
from Bio.PDB import PDBParser
import gzip
import numpy as np
from tqdm import tqdm
import requests
import tarfile
import h5py
from tqdm import tqdm
import pandas as pd
import numpy as np
import time
import json
import h5py
import sys
import os
import shutil
from rdkit import Chem
from rdkit.Chem import Descriptors

# Curate chembl data for all species and store in a folder system

def download_and_extract_chembl(url="https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/latest/chembl_33_sqlite.tar.gz", target_path="data/curated_chembl/"):
    if not os.path.exists("data/curated_chembl"):
        os.makedirs("data/curated_chembl")

    
    if "chembl_33.db" in os.listdir(target_path):
        print("Already extracted chemble database. Skipping...")
        return
    # Download the file from the given URL
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        total_size_in_bytes = int(response.headers.get('content-length', 0))
        chunk_size = 1024 # 1 Kilobyte
        tar_file_path = os.path.join(target_path, 'chembl.tar.gz')
        
        # Save the tar.gz file with a progress bar
        with open(tar_file_path, 'wb') as file, tqdm(
            desc=tar_file_path,
            total=total_size_in_bytes,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(chunk_size=chunk_size):
                size = file.write(data)
                bar.update(size)

        # Extract the tar.gz file
        with tarfile.open(tar_file_path, "r:gz") as tar:
            tar.extractall(path=target_path)
        
        # Clean up the downloaded tar.gz file
        os.remove(tar_file_path)
        print(f"ChEMBL database extracted to {target_path}")
    else:
        print("Failed to download the file")
    
        
    source_path = os.path.join(target_path, "chembl_33", "chembl_33_sqlite", "chembl_33.db")
    destination_path = os.path.join(target_path, "chembl_33.db")

    if os.path.isfile(source_path):
        os.rename(source_path, destination_path)
    else:
        print(f"The file {source_path} does not exist.")

    shutil.rmtree(os.path.join(target_path, "chembl_33", "chembl_33_sqlite"))
    shutil.rmtree(os.path.join(target_path, "chembl_33"))

    print("ChEMBL database extracted to {target_path}")


def extract_af_protein_graph(arg_tuple):
    input_folder_path, output_folder_path, af_uniprot_id = arg_tuple
    input_file_name = os.path.join(input_folder_path, f"AF-{af_uniprot_id}-F1-model_v4.pdb.gz")
    output_file_name = os.path.join(output_folder_path, f"{af_uniprot_id}_graph.h5")
    
    with gzip.open(input_file_name, 'rt') as file:
        parser = PDBParser(QUIET=True)
        atom_structure = parser.get_structure("", file)

    protein_atom_cloud_array = []
    for model in atom_structure:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    atom_type_numeric = graph_tools_pytorch.get_atom_type_numeric(atom.element.strip())
                    atom_data = [atom_type_numeric, *atom.get_coord()]
                    protein_atom_cloud_array.append(atom_data)
    
    protein_atom_cloud_array = np.array(protein_atom_cloud_array)
    atom_point_cloud_atom_types = protein_atom_cloud_array[:, 0]  # Changed from :1 to 0 for correct indexing
    n_atom_types = len(graph_tools_pytorch.atom_type_to_float.values())

    # One-hot encode the atom types
    features = np.eye(n_atom_types)[atom_point_cloud_atom_types.astype(int)]

    # Create the graph
    graph = graph_tools_pytorch.csr_graph_from_point_cloud(protein_atom_cloud_array[:, 1:], STANDARD_BOND_LENGTH=1.5)

    # Convert CSR graph to PyTorch Geometric format
    edge_index = np.vstack(graph.nonzero())
    edge_attr = graph.data

    # Save the graph in h5 file
    with h5py.File(output_file_name, 'w') as h5file:
        h5file.create_dataset('edge_index', data=edge_index)
        h5file.create_dataset('edge_attr', data=edge_attr)
        h5file.create_dataset('atom_features', data=features)  # Save atom types as features

def create_PROTEIN_graphs(input_folder_path="data/curated_chembl/alpha_fold_data/", output_folder_path="data/curated_chembl/af_protein_1_dot_5_angstrom_graphs/"):
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)
    else:
        print(f"Folder {output_folder_path} already exists. Skipping...")
        return

    arg_tuples = []
    af_uniprots = alpha_fold_tools.get_alphafold_uniprot_ids()
    for af_uniprot_id in af_uniprots:
        arg_tuples.append((input_folder_path, output_folder_path, af_uniprot_id))

    with ProcessPoolExecutor() as executor:
        _results = list(tqdm(executor.map(extract_af_protein_graph, arg_tuples), desc="Creating protein atom clouds", total=len(arg_tuples)))


def create_SMILES_id_mappings(chembl_db_path, target_output_path="data/curated_chembl/"):
    if os.path.exists(os.path.join(target_output_path, "molecule_id_mappings", "id_to_smiles.json")):
        print("SMILES id mappings already created. Skipping...")
        return

    conn = sqlite3.connect(chembl_db_path)
    cursor = conn.cursor()

    # Execute a query (example: selecting 10 rows from a table)
    query = "SELECT * FROM bioactivities;"
    cursor.execute(query)

    # Fetch and print the results
    rows = cursor.fetchall()
    idx=0
    id_to_smiles = {}
    smiles_to_id = {}

    if not os.path.exists(os.path.join(target_output_path, "molecule_id_mappings")):
        os.makedirs(os.path.join(target_output_path, "molecule_id_mappings"))

    print("Creating id mappings...")
    for row in rows:
        smiles = row[0]
        id_to_smiles[idx] = smiles
        smiles_to_id[smiles] = idx
        idx += 1

    conn.close()

    print("Saving id mappings...")
    with open(os.path.join(target_output_path, "molecule_id_mappings", "id_to_smiles.json"), 'w') as f:
        json.dump(id_to_smiles, f)

    with open(os.path.join(target_output_path, "molecule_id_mappings", "smiles_to_id.json"), 'w') as f:
        json.dump(smiles_to_id, f)


def generate_and_save_graphs(args, retry=True):
    save_paths = []

    try:
        # Try Generating Multiple Graphs for a 
        smiles, mol_id, target_output_path, id_to_paths, smiles_to_paths, folder_idx = args
        generic_save_path = os.path.join(target_output_path, 'molecule_graphs', f'{folder_idx}', f'{mol_id}_{0}.h5')
        if os.path.exists(generic_save_path):
            return
        
        feature_list, csr_matrix_list = graph_tools_pytorch.get_raw_graphs_from_smiles_string(smiles, num_conformations=5)

        for i, (features, csr_matrix) in enumerate(zip(feature_list, csr_matrix_list)):
            save_path = os.path.join(target_output_path, 'molecule_graphs', f'{folder_idx}', f'{mol_id}_{i}.h5')
            if not os.path.exists(os.path.join(target_output_path, 'molecule_graphs', f'{folder_idx}')):
                os.makedirs(os.path.join(target_output_path, 'molecule_graphs', f'{folder_idx}'))

            # Save each graph in h5 file format
            graph_tools_pytorch.save_features_csr_matrix_to_hdf5(features, csr_matrix, save_path)

            save_paths.append(save_path)

        # Create id mappings for each graph
        id_to_paths[mol_id] = save_paths
        smiles_to_paths[smiles] = save_paths
    except Exception as e:
        print(f"Error occurred while generating graphs: {e}")
        for save_path in save_paths:
            os.remove(save_path)
        if retry:
            print("Retrying...")
            generate_and_save_graphs(args, retry=False)

def create_SMILES_graphs(target_output_path):
    with open(os.path.join(target_output_path, "molecule_id_mappings", "id_to_smiles.json"), 'r') as f:
        id_to_smiles = json.load(f)

    with open(os.path.join(target_output_path, "molecule_id_mappings", "smiles_to_id.json"), 'r') as f:
        smiles_to_id = json.load(f)

    id_to_paths = {}
    smiles_to_paths = {}
    folder_idx = 0
    batch_size = 100000
    batches_per_folder = 1
    batches_in_current_folder = 0
    smiles_batch = []
    mol_id = 0
    batch_mol_ids = []
    for smiles in tqdm(id_to_smiles.values(), desc="Creating SMILES graphs"):
        smiles_batch.append(smiles)        
        batch_mol_ids.append(mol_id)
        mol_id += 1

        if len(smiles_batch)  > batch_size:

            ## Process batch

            # Generate path indexes
            with ProcessPoolExecutor() as executor:
                
                args = [(smiles, mol_id, target_output_path, id_to_paths, smiles_to_paths, folder_idx) for (smiles, mol_id) in zip(smiles_batch, batch_mol_ids)]
                results = list(tqdm(executor.map(generate_and_save_graphs, args), desc="Generating graphs", total=len(smiles_batch)))
            
            # Housekeeping
            batches_in_current_folder += 1
            smiles_batch = []
            batch_mol_ids = []

        if batches_in_current_folder > batches_per_folder:
            folder_idx += 1
            batches_in_current_folder = 0

    with open(os.path.join(target_output_path, "molecule_id_mappings", "id_to_path.json"), 'w') as f:
        json.dump(id_to_paths, f)
    
    with open(os.path.join(target_output_path, "molecule_id_mappings", "smiles_to_path.json"), 'w') as f:
        json.dump(smiles_to_paths, f)


def calculate_descriptors(smiles_string):
    # Convert the SMILES string to a molecule object
    molecule = Chem.MolFromSmiles(smiles_string)

    # Calculate all available descriptors
    descriptors = {}
    for descriptor_name, descriptor_fn in Descriptors.descList:
        try:
            descriptors[descriptor_name] = descriptor_fn(molecule)
        except:
            descriptors[descriptor_name] = None

    return descriptors

def create_SMILES_metadata(target_output_path):
    with open(os.path.join(target_output_path, "molecule_id_mappings", "id_to_smiles.json"), 'r') as f:
        id_to_smiles = json.load(f)
    
    for smiles in tqdm(id_to_smiles.values(), desc="Creating SMILES metadata"):
        descriptors = calculate_descriptors(smiles)
        print(descriptors)


def create_PROTEIN_sequences(alphafold_folder_path, target_output_path): # Update this to make it work
    # Get list of Uniprot IDs
    uniprot_ids = os.listdir(alphafold_folder_path)

    # For each Uniprot ID, read the sequence.fasta file and write to a new file in the target_output_path
    for uniprot_id in uniprot_ids:
        sequence_file_path = os.path.join(alphafold_folder_path, uniprot_id, 'sequence.fasta')
        if os.path.exists(sequence_file_path):
            with open(sequence_file_path, 'r') as f_in:
                sequence = f_in.read()
            with open(os.path.join(target_output_path, f'{uniprot_id}_sequence.fasta'), 'w') as f_out:
                f_out.write(sequence)

def filter_tid(tid):
    if id_mapping_tools.tid_to_af_uniprot(tid):
        return True
    return False

def curate_raw_chembl(raw_chembl_db_path, curated_chembl_db_folder_path, new_db_name="smiles_alphafold_v4_human_uniprot_chembl_bioactivities.db"):
    if os.path.exists(os.path.join(curated_chembl_db_folder_path, new_db_name)):
        print(f"Database {new_db_name} already exists. Skipping...")
        return
    new_db_path = os.path.join(curated_chembl_db_folder_path, new_db_name)

    if not os.path.exists(curated_chembl_db_folder_path):
        os.makedirs(curated_chembl_db_folder_path)

    # Connect to the original database
    conn_original = sqlite3.connect(raw_chembl_db_path)
    cursor_original = conn_original.cursor()

    # Connect to the new database (this will create the file if it doesn't exist)
    conn_new = sqlite3.connect(new_db_path)
    cursor_new = conn_new.cursor()

    # Create a new table in the new database
    cursor_new.execute("""
    CREATE TABLE IF NOT EXISTS bioactivities (
        canonical_smiles VARCHAR(4000),
        af_4_human_uniprot VARCHAR(10),
        activity_type VARCHAR(250),
        activity_relation VARCHAR(50),
        activity_value NUMERIC,
        activity_unit VARCHAR(100)
    );
    """)

    total_rows = 6749321  # Adjust based on the total number of rows

    for x in tqdm(range(0, total_rows, 2000000), desc="loading"):
        query = f"""
        SELECT compound_structures.canonical_smiles, assays.tid, activities.standard_type, activities.standard_relation, activities.standard_value, activities.standard_units
        FROM activities 
        JOIN assays ON activities.assay_id = assays.assay_id
        JOIN compound_structures ON activities.molregno = compound_structures.molregno
        JOIN compound_properties ON activities.molregno = compound_properties.molregno
        WHERE assays.assay_organism = 'Homo sapiens'
        LIMIT 2000000 OFFSET {x};
        """
        cursor_original.execute(query)
        rows = cursor_original.fetchall()
        
        for row in rows:
            if filter_tid(row[1]):  # Replace tid_column_index with the index of tid in the row
                new_row = list(row)
                new_row[1] = id_mapping_tools.tid_to_af_uniprot(new_row[1])

                cursor_new.execute("INSERT INTO bioactivities VALUES (?, ?, ?, ?, ?, ?)", tuple(new_row))  # Adjust the placeholders based on the number of columns

    # Commit changes to the new database and close both connections
    conn_new.commit()
    conn_original.close()
    conn_new.close()


def download_file(url, target_path, chunk_size=1024):
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    with open(target_path, 'wb') as file, tqdm(
        desc=target_path,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=chunk_size):
            size = file.write(data)
            bar.update(size)

def extract_tar_file(file_path, target_dir):
    try:
        with tarfile.open(file_path, 'r') as tar, tqdm(
            desc="Extracting",
            total=len(tar.getmembers()),
            unit='file'
        ) as bar:
            for member in tar:
                tar.extract(member, path=target_dir)
                bar.update(1)
    except Exception as e:
        print(f"Error occurred while extracting: {e}")

def download_and_extract(url, target_dir):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    file_name = url.split('/')[-1]
    file_path = os.path.join(target_dir, file_name)
    
    if not os.path.exists(file_path):
        print(f"Downloading {file_name}...")
        download_file(url, file_path)
    else:
        print("File {file_name} already exists...")

    print(f"Extracting {file_name}...")
    if len(os.listdir(target_dir)) > 1:
        print("Folder not empty. Skipping extraction...")
    else:
        print("Folder empty. Extracting...")
        extract_tar_file(file_path, target_dir)

def download_alphafold_data(target_output_path="data/curated_chembl/alpha_fold_data"):
    url = "https://ftp.ebi.ac.uk/pub/databases/alphafold/latest/UP000005640_9606_HUMAN_v4.tar"
    download_and_extract(url, target_output_path)


def get_ordered_unique_col_values(columns, af_uniprots=alpha_fold_tools.get_alphafold_uniprot_ids(), id_mappings=pd.read_table("molytica_m/data_tools/idmapping_af_uniprot_metadata.tsv")):
    values = values = [set() for _ in range(len(columns))]
    print(values)

    for uniprot in tqdm(af_uniprots, desc="Getting unique values"):
        row = id_mappings[id_mappings['From'] == str(uniprot)].iloc[0]

        for col in columns:
            col_id = list(columns).index(col)
            vals = str(row[col]).split("; ")
            for val in vals:
                values[col_id].add(val)
    
    unique_value_lists_dict = {}
    for idx, value in enumerate(values):
        unique_value_lists_dict[list(columns)[idx]] = sorted(list(value))
    return unique_value_lists_dict


def binary_encode(full_values_list, values):
    value_to_index = {v: i for i, v in enumerate(full_values_list)}
    binary_encoded = [0] * len(full_values_list)
    for value in values:
        if value in value_to_index:
            binary_encoded[value_to_index[value]] = 1
    return binary_encoded

def create_PROTEIN_metadata(save_path="data/curated_chembl/af_metadata/"): # AF-UNIPROT-HOMO-SAPEINS protein metadata as binary vectors
    print("Creating protein")
    id_mappings = pd.read_table("molytica_m/data_tools/idmapping_af_uniprot_metadata.tsv")
    af_uniprots = alpha_fold_tools.get_alphafold_uniprot_ids()

    if len(save_path) > 0:
        print("Protein Metadata already created. Skipping.")
        return

    columns = id_mappings.columns

    unique_value_lists = get_ordered_unique_col_values(columns)

    interesting_columns = ["Gene Ontology (GO)"]
    uniprot_metadata = {}

    for uniprot in tqdm(af_uniprots, desc="Generating Metadata files"):
        uniprot_data = []
        row = id_mappings[id_mappings['From'] == str(uniprot)].iloc[0]

        for col in interesting_columns:
            vals = list(set(str(row[col]).split("; ")))
            uniprot_data += binary_encode(unique_value_lists[col], vals)

        uniprot_metadata[uniprot] = uniprot_data

    if not os.path.exists(save_path):
        os.makedirs(save_path)


    for uniprot in tqdm(list(uniprot_metadata.keys()), desc="Saving to hdf5"):
        metadata = uniprot_metadata[uniprot]

        file_name = os.path.join(save_path, f"{uniprot}_metadata.h5")

        # Saving metadata to HDF5 file
        with h5py.File(file_name, 'w') as h5file:
            h5file.create_dataset('metadata', data=np.array(metadata, dtype=float))

def main():
    alphafold_folder_path = "data/curated_chembl/alpha_fold_data"
    raw_chembl_db_path = "data/curated_chembl/chembl_33.db"
    curated_chembl_db_folder_path = "data/curated_chembl/"
    new_db_name = 'smiles_alphafold_v4_human_uniprot_chembl_bioactivities.db'
    protein_graph_output_path = "data/curated_chembl/af_protein_1_dot_5_angstrom_graphs"
    protein_metadata_tsv_path = "/path/to/protein/metadata/tsv" # Curated metadata created by pasting all the uniprot ids into uniprot website id mapping tool
    target_output_path = "data/curated_chembl"
    target_protein_metadata_output_path = "data/curated_chembl/af_metadata"
    curated_chembl_db_path = os.path.join(curated_chembl_db_folder_path, new_db_name)

    download_and_extract_chembl()

    id_mapping_tools.generate_index_dictionaries(raw_chembl_db_path)

    download_alphafold_data(alphafold_folder_path)
    curate_raw_chembl(raw_chembl_db_path, curated_chembl_db_folder_path, new_db_name)
    create_PROTEIN_graphs(alphafold_folder_path, protein_graph_output_path)
    create_PROTEIN_metadata(target_protein_metadata_output_path)

    create_SMILES_id_mappings(curated_chembl_db_path, target_output_path)
    create_SMILES_graphs(target_output_path)
    
    create_SMILES_metadata(target_output_path)
    create_PROTEIN_sequences(alphafold_folder_path, target_output_path)

    # Molecule sequences are represented as smile strings


if __name__ == "__main__":
    main()
