import os

def get_alphafold_uniprot_ids():
    ids = [x.split("-")[1] for x in os.listdir("data/alpha_fold_data/") if type(x) == str and len(x.split("-")) > 1]
    return sorted(list(set(ids)))