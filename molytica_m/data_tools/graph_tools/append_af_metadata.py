from molytica_m.data_tools import alpha_fold_tools
from tqdm import tqdm
import pandas as pd
import numpy as np
import time
import json
import h5py
import sys
import os

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

def curate_af_human_uniprot_metadata(save_path="data/af_metadata/"):
    id_mappings = pd.read_table("molytica_m/data_tools/idmapping_af_uniprot_metadata.tsv")
    af_uniprots = alpha_fold_tools.get_alphafold_uniprot_ids()

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

if __name__ == "__main__":
    curate_af_human_uniprot_metadata()