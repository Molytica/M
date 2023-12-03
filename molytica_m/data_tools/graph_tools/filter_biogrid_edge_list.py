import json

with open("molytica_m/data_tools/uniprot_edges_biogrid.json", "r") as file:
    edges = json.load(file)["uniprot_edges_biogrid"]



print(len(edges))
print(edges[:200])

filtered_edges = []

for edge in edges:
    if None not in edge: 
        new_edge = (edge[0], edge[1])
        filtered_edges.append(new_edge)
        print(f"Edge is not None {edge}")
    else:
        print(f"Edge is None {edge}")

with open("molytica_m/data_tools/uniprot_edges_biogrid_filtered.json", "w") as file:
    json_data = {"uniprot_edges_biogrid_filtered": filtered_edges}
    json.dump(json_data, file)