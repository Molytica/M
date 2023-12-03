from molytica_m.data_tools import alpha_fold_tools
from tqdm import tqdm
import pandas as pd
import json

id_mappings = pd.read_table("molytica_m/data_tools/idmapping_af_uniprot_metadata.tsv")
af_uniprots = alpha_fold_tools.get_alphafold_uniprot_ids()

columns = id_mappings.columns
print(columns)


def get_ordered_unique_col_values(column):
    values = set()
    for uniprot in tqdm(af_uniprots, desc="Getting unique values"):
        row = id_mappings[id_mappings['From'] == str(uniprot)].iloc[0]
        values.add(str(row[column]))
    return sorted(list(values))

af_metadata = {}



print(get_ordered_unique_col_values("Gene Ontology (biological process)"))



