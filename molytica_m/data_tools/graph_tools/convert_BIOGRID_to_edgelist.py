from concurrent.futures import ProcessPoolExecutor
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
    try:
        # URL to fetch
        url = f"https://thebiogrid.org/{biogrid_id}/summary/homo-sapiens/"
        
        # Get the HTML content with a timeout set (e.g., 10 seconds)
        response = requests.get(url, timeout=10)
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

    except requests.exceptions.Timeout:
        # If a timeout occurs, return None
        return None
    except requests.exceptions.RequestException as e:
        # For any other exceptions, print the exception and return None
        print(e)
        return None


def process_edge(edge):
    edge = [str(val) for val in edge]
    
    node1_uniprot_id = get_uniprot_from_biogrid_id(edge[0])
    node2_uniprot_id = get_uniprot_from_biogrid_id(edge[1])
    
    return edge, (node1_uniprot_id, node2_uniprot_id)

# Load processed edges if they exist
edges_processed = []
if os.path.exists("molytica_m/data_tools/edges_processed_biogrid.json"):
    with open("molytica_m/data_tools/edges_processed_biogrid.json", "r") as file:
        edges_processed = json.load(file)["edges_processed_biogrid"]

uniprot_edges = []
if os.path.exists("molytica_m/data_tools/uniprot_edges_biogrid.json"):
    with open("molytica_m/data_tools/uniprot_edges_biogrid.json", "r") as file:
        uniprot_edges = json.load(file)["uniprot_edges_biogrid"]

# Filter out already processed edges
biogrid_id_edge_list = [edge for edge in biogrid_id_edge_list if edge not in edges_processed]

# Use ProcessPoolExecutor to run processes in parallel
with ProcessPoolExecutor() as executor:
    results = list(tqdm(executor.map(process_edge, biogrid_id_edge_list), total=len(biogrid_id_edge_list), desc="Converting Entrez to Uniprot"))

# Update the edges_processed and uniprot_edges lists
for edge, uniprot_edge in results:
    edges_processed.append(edge)
    uniprot_edges.append(uniprot_edge)

# Save the updated lists to the respective JSON files
with open("molytica_m/data_tools/uniprot_edges_biogrid.json", "w") as file:
    json.dump({"uniprot_edges_biogrid": uniprot_edges}, file)

with open("molytica_m/data_tools/edges_processed_biogrid.json", "w") as file:
    json.dump({"edges_processed_biogrid": edges_processed}, file)