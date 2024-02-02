import os

loaded = False
alpha_fold_data = None

def generate_alpha_fold_data(af_data_folder_path="data/curated_chembl/alpha_fold_data"):
    if os.path.exists(af_data_folder_path):
        ids = [x.split("-")[1] for x in os.listdir(af_data_folder_path) if type(x) == str and len(x.split("-")) > 1]
        return sorted(list(set(ids)))
    return None

def get_all_alphafold_uniprot_ids(af_data_folder_path="data/curated_chembl/alpha_fold_data/"):
    af_uniprots = {}
    for folder in os.listdir(af_data_folder_path):
        if not os.path.isdir(os.path.join(af_data_folder_path, folder)):
            continue
        af_uniprots[folder] = set()
    
        for file_name in os.listdir(os.path.join(af_data_folder_path, folder)):
            if "AF-" not in file_name:
                continue
            af_uniprot = file_name.split("-")[1]
            af_uniprots[folder].add(af_uniprot)

        af_uniprots[folder] = sorted(list(af_uniprots[folder]))
    
    return af_uniprots


def get_alphafold_uniprot_ids(af_data_folder_path="data/curated_chembl/alpha_fold_data"):
    return get_all_alphafold_uniprot_ids()["HUMAN"]


if __name__ == "__main__":
    dict_af = get_all_alphafold_uniprot_ids()
    print(dict_af.keys())
    print(len(dict_af["YEAST"]))