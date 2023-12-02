from concurrent.futures import ProcessPoolExecutor
from molytica_m.data_tools import alpha_fold_tools
from tqdm import tqdm
import numpy as np
import json

with open("molytica_m/research/iP_result_tuples.json", "r") as file:
    iP_result_tuples_list = json.load(file)["iP_result_tuples"]

def get_score(tuple):
    uniprot, smiles = tuple
    for x in iP_result_tuples_list:
        if x[0][0] == uniprot and x[0][1] == smiles:
            return x[1]
    return 0

smiless = sorted(list(set([x[0][1] for x in iP_result_tuples_list])))

uniprots = alpha_fold_tools.get_alphafold_uniprot_ids()
score_matrix = np.zeros((len(uniprots), len(smiless)))

tuples_to_iterate = []

for uniprot_idx, uniprot in tqdm(enumerate(uniprots), desc="Loading", total=len(uniprots)):
    for smiles_idx, smiles in enumerate(smiless):
        tuples_to_iterate.append((uniprot, smiles))
        
with ProcessPoolExecutor() as executor:
        scores = list(tqdm(executor.map(get_score, tuples_to_iterate), desc="Getting scores", total=len(tuples_to_iterate))) 

print(score_matrix)
print(np.count_nonzero(score_matrix))