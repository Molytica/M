from molytica_m.data_tools.alpha_fold_tools import get_alphafold_uniprot_ids
from molytica_m.data_tools.graph_tools import graph_tools, interactome_tools
import random
import json

af_uniprots = get_alphafold_uniprot_ids()
save_path = "data/iPPI_data"
molecule_smiles = "molytica_m/data_tools/DLiP_rule_of_5_compound_data.json"
DLiP_data = {}

with open(molecule_smiles, "r") as file:
    DLiP_data = json.load(file)

edge_list = [x["proteins"] for x in DLiP_data.values()]

# Create a graph of all the interactions. And every cluster (interacting proteins) create a set of the uniprot_ids
sub_graph_nodes = graph_tools.get_subgraph_nodes_from_edgelist(edge_list)
random.shuffle(sub_graph_nodes)

train_val_split = int(len(sub_graph_nodes) * 0.8)
val_test_split = int(len(sub_graph_nodes) * 0.9)

nodes = {"train": sub_graph_nodes[:train_val_split],
            "val": sub_graph_nodes[train_val_split:val_test_split],
            "test": sub_graph_nodes[val_test_split:],}

uniprots = {}
DLiP_ids = {}
for key in nodes.keys():
    DLiP_ids[key] = interactome_tools.get_DLiP_ids_from_nodes(nodes[key], DLiP_data)
    uniprots[key] = interactome_tools.get_uniprots(DLiP_ids[key], DLiP_data)


print(set(DLiP_ids["train"]) & set(DLiP_ids["val"]) & set(DLiP_ids["test"])) # Check for DLiP ids that are on multiple sets
print(len(set(DLiP_ids["train"]) | set(DLiP_ids["val"]) | set(DLiP_ids["test"]))) # Check number of DLiP ids in total
print(set(uniprots["train"]) & set(uniprots["val"]) & set(uniprots["test"])) # Check for overlapping prot ids

# DLiP ids are successfully split if program returns set(), 7021, set()
# Inflate datasets with equal amount of non interacting

# Assume if a molecule is not associated with a pair, it does not inhibit that pair.[Assumption] to create negative training data
# Fill in with equal amount of assumed non iPPI prot pairs + mol

for key in nodes.keys():
    PPI_molecules = interactome_tools.get_PPI_molecules(DLiP_ids[key], DLiP_data, af_uniprots)
    graph_tools.save_graphs(PPI_molecules, save_path, key)
