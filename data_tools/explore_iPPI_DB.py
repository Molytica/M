import json

with open("data_tools/iPPI-DB.json", "r") as file:
    json_data = json.load(file)

activity_types = set()
for value in json_data.values():
    for PPI_entry in value["PPI_VALUES"]:
        activity_type = PPI_entry["activity_type"]
        activity_types.add(activity_type)
        

print(activity_types)