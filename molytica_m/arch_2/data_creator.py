import os

# Curate chembl data for all species and store in a folder system

def download_alphafold_data(target_output_path):
    pass

def create_PROTEIN_graphs(alphafold_folder_path, target_output_path):
    # Perform data creation into the target_output folder
    # Add your code here

    # read the alphafold folder and get the uniprot ids

    # for each uniprot id, read the pdb file and create the graph

    pass



def create_SMILES_graphs_and_id_mappings(chembl_db_path, target_output_path):
    # Perform data creation into the target_output folder
    # Add your code here

    # Create a folder called molecule_graphs inside the target_output folder if not exists
    if os.path.exists(os.path.join(target_output_path, "molecule_graphs")):
        os.makedirs(os.path.join(target_output_path, "molecule_graphs"))

    # Create a folder called id_mappings inside the target_output folder if not exists
    if os.path.exists(os.path.join(target_output_path, "molecule_id_mappings")):
        os.makedirs(os.path.join(target_output_path, "molecule_id_mappings"))

    # Create id mapping file

    # Create graph hdf5 files where the file name is the smiles id followed by .h5

    pass


def create_PROTEIN_metadata(alphafold_folder_path, protein_metadata_tsv_path, target_output_path):
    # Perform data creation into the target_output folder
    # Add your code here

    # Use alphafold folder and protein_metadata_tsv_path to create protein metadata for each uniprot id


    pass



def create_SMILES_metadata(chembl_db_path, target_output_path):
    # Perform data creation into the target_output folder
    # Add your code here

    # Use rdkit to create molecular descriptors (molecule metadata) for each SMILES string

    pass


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


def main():
    alphafold_folder_path = "/path/to/alphafold/folder"
    chembl_db_path = "/path/to/chembl/db"
    protein_metadata_tsv_path = "/path/to/protein/metadata/tsv"
    target_output_path = "/path/to/protein_modulation_model_curated_chembl_data"

    download_alphafold_data(alphafold_folder_path)
    create_PROTEIN_graphs(alphafold_folder_path, target_output_path)
    create_SMILES_graphs_and_id_mappings(chembl_db_path, target_output_path)
    create_PROTEIN_metadata(alphafold_folder_path, protein_metadata_tsv_path, target_output_path)
    create_SMILES_metadata(chembl_db_path, target_output_path)
    create_PROTEIN_sequences(alphafold_folder_path, target_output_path)

    # Molecule sequences are represented as smile strings


if __name__ == "__main__":
    main()
