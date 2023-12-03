from molytica_m.data_tools import alpha_fold_tools
import pandas as pd
import json

id_mappings = pd.read_table("molytica_m/data_tools/idmapping_2023_11_18.tsv")
af_uniprots = alpha_fold_tools.get_alphafold_uniprot_ids()


af_uniprot_metadata = {}

with open("molytica_m/data_tools/af_uniprots_txt.txt", "w") as file:
    for uniprot in af_uniprots:
        file.write(uniprot)
        file.write(" ")


print(id_mappings)