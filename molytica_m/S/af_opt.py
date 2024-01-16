from tqdm import tqdm
from molytica_m.data_tools import alpha_fold_tools
import os
import gzip
import time, re
import numpy as np
import h5py
from molytica_m.S import MPGOA1
from multiprocessing import Pool
from tqdm.contrib.concurrent import process_map  # Import process_map

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
    'Se': 17.0,# Selenium
}

def extract_atom_data_from_pdbgz_text(pdbgz_file):
    atom_data = []

    with gzip.open(pdbgz_file, 'rt') as f:
        for line in f:
            if line.startswith("ATOM"):
                # Use regular expression to extract atom type and coordinates
                cols = line.replace("-", " -").split()

                try:
                    atom_type = np.float32(atom_type_to_float[cols[-1]])
                    x_coord = np.float32(cols[-6])
                    y_coord = np.float32(cols[-5])
                    z_coord = np.float32(cols[-4])

                    atom_data.append([atom_type, x_coord, y_coord, z_coord])
                except Exception as e:
                    print(e)
                    print(line)
    return atom_data



def process_protein(args):
    species, file_name, uniprot_id = args

    file_path = os.path.join("data/curated_chembl/alpha_fold_data", species, file_name)

    atom_data = extract_atom_data_from_pdbgz_text(file_path)

    atom_data = np.array(atom_data)

    atom_data = MPGOA1.get_optimised_atom_type_and_coords(atom_data)

    # Save the atom data
    save_folder = os.path.join("data/curated_chembl/opt_af_coords", species)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    save_path = os.path.join("data/curated_chembl/opt_af_coords", species, uniprot_id + ".h5")

    with h5py.File(save_path, 'w') as h5file:
        h5file.create_dataset('atom_data', data=atom_data) 

def opt_af_coords_for_species(speciess):

    for species in tqdm(speciess, desc="Opt species of interest"):
        
        args = []
        
        for file_name in tqdm(os.listdir(os.path.join("data/curated_chembl/alpha_fold_data", species)), desc="Opt files for {}".format(species)):
            if ".pdb" not in file_name:
                continue
            uniprot_id = file_name.split("-")[1]

            args.append((species, file_name, uniprot_id))
            
        
        process_map(process_protein, args, max_workers=4)




if __name__ == "__main__":
    species = ["HUMAN", "RAT", "MOUSE", "YEAST"]
    opt_af_coords_for_species(species)