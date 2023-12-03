from molytica_m.data_tools import alpha_fold_tools
from tqdm import tqdm
import json

with open("molytica_m/data_tools/uniprot_edges_biogrid_filtered.json", "r") as file:
    biogrid_uniprot_edges = json.load(file)["uniprot_edges_biogrid_filtered"]


with open("molytica_m/data_tools/HuRI_edge_list.json", "r") as file:
    huri_uniprot_edges = json.load(file)["HuRI_edge_list"]



def convert_to_tuple_set(edge_list):
    tuple_set = set()
    for edge in edge_list:
        tuple_set.add((edge[0], edge[1]))
    return tuple_set


biogrid_set = convert_to_tuple_set(biogrid_uniprot_edges)
huri_set = convert_to_tuple_set(huri_uniprot_edges)

overlap = biogrid_set & huri_set
biogrid_but_not_huri = biogrid_set - huri_set
huri_but_not_biogrid = huri_set - biogrid_set

combined_set = list(huri_set | biogrid_set)
print(len(combined_set))
#print(20504**2)
#print(len(combined_set) / (20504**2))

filtered_set = set()
af_uniprots = alpha_fold_tools.get_alphafold_uniprot_ids()

for item in tqdm(combined_set, desc="filtering"):
    if (item[1], item[0]) not in filtered_set and item[0] in af_uniprots and item[1] in af_uniprots:
        filtered_set.add(item)

print(len(filtered_set))

with open("molytica_m/data_tools/filtered_no_reverse_duplicates_huri_and_biogrid_af_uniprot_edges.json", "w") as file:
    json_data = {"filtered_no_reverse_duplicates_huri_and_biogrid_af_uniprot_edges": list(filtered_set)}
    json.dump(json_data, file)
