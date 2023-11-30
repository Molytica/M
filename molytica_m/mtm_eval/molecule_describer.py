from molytica_m.data_tools import gpt_tools
from bs4 import BeautifulSoup, Comment
import pandas as pd
import requests
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

# Initialize 'description' column with empty strings
df_score_sorted['description'] = [''] * df_score_sorted.shape[0]

# Fetch descriptions for only the first ten rows
for idx, row in df_score_sorted.head(10).iterrows():
    cid = search_pubchem_by_smiles(row['Molecule'])  # Replace 'SMILES' with the actual column name
    mol_info = get_molecule_info(cid)
    description = gpt_tools.ask_gpt(str(mol_info), "Describe the function of this molecule based on litterature and your judgement.", "gpt-3.5-turbo-1106", 0.2)
    df_score_sorted.at[idx, 'description'] = description
    print("Done.")

# Now, save the DataFrame to CSV
df_score_sorted.to_csv("molytica_m/mtm_eval/molecule_evaluations.csv")


    


