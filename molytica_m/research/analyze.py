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

for val in tqdm(iP_result_tuples_list):
    uniprot = val[0][0]
    smiles = val[0][1]
    score = val[1]

    score_matrix[uniprots.index(uniprot), smiless.index(smiles)] = score

print(np.count_nonzero(score_matrix))