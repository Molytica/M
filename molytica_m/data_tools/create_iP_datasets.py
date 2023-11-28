from molytica_m.data_tools.alpha_fold_tools import get_alphafold_uniprot_ids
from molytica_m.data_tools.graph_tools import graph_tools
from tqdm import tqdm
import numpy as np
import random
import json

with open("molytica_m/data_tools/iPPI-DB.json", "r") as file:
    json_data = json.load(file)

activity_types = set()
uniprot_ids = set()
activity_types_count = {
    'Kd ratio (Kd without/with ligand': 0,
    'pIC50 (half maximal inhibitory concentration, -log10)': 0,
    'pKd (dissociation constant, -log10)': 0,
    'pKi (inhibition constant, -log10)': 0,
    'pEC50 (half maximal effective concentration, -log10)': 0
}

for value in json_data.values():
    for PPI_entry in value["PPI_VALUES"]:
        activity_type = PPI_entry["activity_type"]
        uniprot_id = PPI_entry["target_protid"]
        uniprot_ids.add(uniprot_id)
        activity_types_count[activity_type] += 1
        activity_types.add(activity_type)

uniprots = sorted(list(uniprot_ids))
random.shuffle(uniprots)
train_cutoff = int(float(len(uniprots)) * 0.8)
val_cutoff = int(float(len(uniprots)) * 0.9)

uniprot_ids_split = {}
for i, uniprot_id in enumerate(uniprots):
    if i < train_cutoff:
        uniprot_ids_split[uniprot_id] = "train"
    elif i < val_cutoff:
        uniprot_ids_split[uniprot_id] = "val"
    else:
        uniprot_ids_split[uniprot_id] = "test"

alphafold_uniprot_ids = get_alphafold_uniprot_ids()
idx = 0

values = list(json_data.values())
random.shuffle(values)
for value in tqdm(values, desc="Generating iP dataset"):
    SMILES = value["SMILES"]

    for PPI in value["PPI_VALUES"]:
        uniprot_id = PPI["target_protid"]
        activity_type = PPI["activity_type"]
        activity = PPI["activity"]

        if activity_type in ["pIC50 (half maximal inhibitory concentration, -log10)", "pKi (inhibition constant, -log10)"] and uniprot_id in alphafold_uniprot_ids:
            # Use this data! Not other types at the moment
            try:
                G = graph_tools.combine_graphs([graph_tools.get_graph_from_uniprot_id(uniprot_id), graph_tools.get_graph_from_smiles_string(SMILES)])
                G.y = np.array([activity], dtype=np.float64)

                graph_tools.save_graph(G, f"data/iP_data/{uniprot_ids_split[uniprot_id]}/iP_{idx}.h5")
            except Exception as e:
                print(e)
        idx += 1

