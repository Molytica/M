from molytica_m.data_tools import alpha_fold_tools
from tqdm import tqdm
import pandas as pd
import time
import json
import sys

id_mappings = pd.read_table("molytica_m/data_tools/idmapping_af_uniprot_metadata.tsv")
af_uniprots = alpha_fold_tools.get_alphafold_uniprot_ids()

columns = id_mappings.columns
print(columns)


def get_ordered_unique_col_values(columns):
    values = values = [set() for _ in range(len(columns))]
    print(values)

    for uniprot in tqdm(af_uniprots, desc="Getting unique values"):
        row = id_mappings[id_mappings['From'] == str(uniprot)].iloc[0]

        for col in columns:
            col_id = list(columns).index(col)
            vals = str(row[col]).split("; ")
            for val in vals:
                values[col_id].add(val)
    
    unique_value_lists_dict = {}
    for idx, value in enumerate(values):
        unique_value_lists_dict[list(columns)[idx]] = sorted(list(value))
    return unique_value_lists_dict

af_metadata = {}

unique_value_lists = get_ordered_unique_col_values(columns)

print(unique_value_lists[])



