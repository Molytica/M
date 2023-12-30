import os

loaded = False
alpha_fold_data = None

def generate_alpha_fold_data(af_data_folder_path="data/curated_chembl/alpha_fold_data"):
    if os.path.exists(af_data_folder_path):
        ids = [x.split("-")[1] for x in os.listdir(af_data_folder_path) if type(x) == str and len(x.split("-")) > 1]
        return sorted(list(set(ids)))
    return None

def get_alphafold_uniprot_ids(af_data_folder_path="data/curated_chembl/alpha_fold_data"):
    global loaded
    global alpha_fold_data  # Declare alpha_fold_data as global
    if not loaded:
        alpha_fold_data = generate_alpha_fold_data(af_data_folder_path)
        if alpha_fold_data is None:
            print("AlphaFold data not found")
        else:
            loaded = True
    return alpha_fold_data