import pandas as pd
import requests
from bs4 import BeautifulSoup, Comment
import time

df = pd.read_csv("molytica_m/mtm_eval/molecule_evaluations.csv")

def get_visible_text(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')

    # Remove script and style elements
    for script_or_style in soup(["script", "style"]):
        script_or_style.decompose()

    # Remove comments
    for comment in soup.find_all(text=lambda text: isinstance(text, Comment)):
        comment.extract()

    # Get text
    text = soup.get_text()

    # Break into lines and remove leading/trailing whitespace
    lines = (line.strip() for line in text.splitlines())
    # Break multi-headlines into a line each
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    # Drop blank lines
    text = '\n'.join(chunk for chunk in chunks if chunk)

    return text

def get_molecule_info(cid):
    base_url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
    url = f"{base_url}/compound/cid/{cid}/record/JSON"
    
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        return data
    else:
        return "Error: Unable to fetch data for CID {}".format(cid)

def search_pubchem_by_smiles(smiles):
    base_url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
    url = f"{base_url}/compound/fastidentity/smiles/{smiles}/cids/JSON"

    response = requests.get(url)
    if response.status_code == 200:
        try:
            return response.json()['IdentifierList']['CID'][0]
        except Exception as e:
            print(e)
            print(f"Could not get cid of smiles {smiles}")
    
    return f"Error: {response.status_code}, unable to fetch data for SMILES {smiles}"

df_score_sorted = df.sort_values(by='Score', ascending=False)

info_column = {}

for idx, row in df_score_sorted.iterrows():
    smiles_data = {"smiles": row[2], "id": row[0], "smiles_score": row[3], "description": get_molecule_info(search_pubchem_by_smiles(row[2]))}
    print(smiles_data["description"])


