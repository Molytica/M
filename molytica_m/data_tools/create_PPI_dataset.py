from molytica_m.data_tools import alpha_fold_tools
from collections import Counter
import itertools
import random
import json


af_uniprots = alpha_fold_tools.get_alphafold_uniprot_ids()

with open("molytica_m/data_tools/filtered_no_reverse_duplicates_huri_and_biogrid_uniprot_edges.json", "r") as file:
    edge_list = json.load(file)["filtered_no_reverse_duplicates_huri_and_biogrid_uniprot_edges"]

random.shuffle(edge_list)

n_edges = len(edge_list)
split = []
train_frac = 0.8
val_frac = 0.1

for idx, edge in enumerate(edge_list):
    if idx / n_edges < train_frac:
        split.append("train")
    elif idx / n_edges < train_frac + val_frac:
        split.append("val")
    else:
        split.append("test")

"""
print(n_edges)
print(len(split))

occurrences = Counter(split)

for occ in occurrences.values():
    print(occ / n_edges)

"""



# Full PPI
theoretical_edge_list = list(itertools.combinations_with_replacement(af_uniprots, 2))
print(len(theoretical_edge_list))

# PPI pos + equal number PPI neg (half positive half negative)