import os

loaded = False
alpha_fold_data = None

def generate_alpha_fold_data(af_data_folder_path="data/curated_chembl/alpha_fold_data"):
    ids = [x.split("-")[1] for x in os.listdir(af_data_folder_path) if type(x) == str and len(x.split("-")) > 1]
    return sorted(list(set(ids)))

def get_alphafold_uniprot_ids(af_data_folder_path="data/curated_chembl/alpha_fold_data"):
    if not loaded:
        alpha_fold_data = generate_alpha_fold_data(af_data_folder_path)
        loaded = True
    return alpha_fold_data