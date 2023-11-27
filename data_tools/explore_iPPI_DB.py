import json

with open("data_tools/iPPI-DB.json", "r") as file:
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



for value in json_data.values():
    SMILES = value["SMILES"]
    print(SMILES)

    for PPI in value["PPI_VALUES"]:
        uniprot_id = PPI["target_protid"]
        activity_type = PPI["activity_type"]
        activity = PPI["activity"]
        print(f"Debug Info - Uniprot ID: {uniprot_id}, Activity Type: {activity_type}, Activity: {activity}")
    
    print()


print(activity_types_count)
print(len(uniprot_ids))