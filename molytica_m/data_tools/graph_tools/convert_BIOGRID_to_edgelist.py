from bs4 import BeautifulSoup
from tqdm import tqdm
import pandas as pd
import requests
import json
import os

df = pd.read_table("data/uzipped_biogrid/BIOGRID-ORGANISM-Homo_sapiens-4.4.227.tab3.txt")

subdf = df[["BioGRID ID Interactor A", "BioGRID ID Interactor B"]]
biogrid_id_edge_list = list(set(tuple(x) for x in subdf.values))

print(df.head())

def get_uniprot_from_biogrid_id(biogrid_id):
    # URL to fetch
    url = f"https://thebiogrid.org/{biogrid_id}/summary/homo-sapiens/"
    
    # Get the HTML content
    response = requests.get(url)
    html = response.text

    # Parse the HTML with BeautifulSoup
    soup = BeautifulSoup(html, 'html.parser')

    # Find all links
    links = soup.find_all('a')

    # Loop through links and extract URLs or specific content
    for link in links:
        href = link.get('href')
        if 'uniprot' in href:
            # Extract relevant part or do further processing
            return href.split("/")[-1]


edges_processed = []
if os.path.exists("molytica_m/data_tools/edges_processed_biogrid.json"):
    with open("molytica_m/data_tools/edges_processed_biogrid.json", "r") as file:
        edges_processed = json.load(file)["edges_processed_biogrid"]


uniprot_edges = list()
if os.path.exists("molytica_m/data_tools/uniprot_edges_biogrid.json"):
    with open("molytica_m/data_tools/uniprot_edges_biogrid.json", "r") as file:
        uniprot_edges = json.load(file)["uniprot_edges_biogrid"]

for edge in tqdm(biogrid_id_edge_list, desc="Converting Entrez to Uniprot"):
    edge = [str(val) for val in edge]
    if edge in edges_processed:
        continue

    node1_uniprot_id = get_uniprot_from_biogrid_id(edge[0])
    node2_uniprot_id = get_uniprot_from_biogrid_id(edge[1])

    uniprot_edges.append((node1_uniprot_id, node2_uniprot_id))
    edges_processed.append(edge)

    with open("molytica_m/data_tools/uniprot_edges_biogrid.json", "w") as file:
        json_data = {"uniprot_edges_biogrid": list(uniprot_edges)}
        json.dump(json_data, file)

    with open("molytica_m/data_tools/edges_processed_biogrid.json", "w") as file:
        json_data = {"edges_processed_biogrid": list(edges_processed)}
        json.dump(json_data, file)