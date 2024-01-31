# Load model directly
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch
import multiprocessing
from tqdm import tqdm
import os
import h5py
from rdkit import Chem
from rdkit.Chem import inchi
from multiprocessing import Process, Lock
import json

def get_chemBERT_tok_mod(device="cpu"):
    # Check if GPU is available and set the device accordingly
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #print("Using device:", "cuda" if torch.cuda.is_available() else "cpu")
    if device == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device:", "cuda" if torch.cuda.is_available() else "cpu")
    #print("Using device in chembert script:", "cpu")

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("DeepChem/ChemBERTa-77M-MLM")
    model = AutoModelForMaskedLM.from_pretrained("DeepChem/ChemBERTa-77M-MLM")
    model.to(device)
    model.share_memory()

    return tokenizer, model


def get_molecule_mean_logits(smiles, tokenizer, model):
    smiles_input = smiles[:512]
    # Tokenize the input SMILES string
    with torch.no_grad():
        inputs = tokenizer(smiles_input, return_tensors="pt").to(model.device)

        # Perform inference with the model
        outputs = model(**inputs).logits.mean(dim=1, keepdim=True)

        return outputs
    
def get_molecule_logits(smiles, tokenizer, model):
    smiles_input = smiles[:512]
    # Tokenize the input SMILES string
    with torch.no_grad():
        inputs = tokenizer(smiles_input, return_tensors="pt").to(model.device)

        # Perform inference with the model without taking the mean
        outputs = model(**inputs).logits

        return outputs

def process_1():
    # Import necessary modules and define functions like get_chemBERT_tok_mod and get_molecule_mean_logits
    # ...

    print("In process 1")
    smiles_1 = "CC(=O)OC1=CC=CC=C1C(=O)O"
    tok1, mod1 = get_chemBERT_tok_mod()
    shape_1 = get_molecule_mean_logits(smiles_1, tok1, mod1)
    print(shape_1[0][0].shape)
    print("Done with process 1")

def process_2():
    # Import necessary modules and define functions like get_chemBERT_tok_mod and get_molecule_mean_logits
    # ...
    print("In process 2")
    smiles_1 = "CC(=O)OC1==O)OC1=CC=CC=CC=CC=C1C(=O)O"
    tok2, mod2 = get_chemBERT_tok_mod()
    shape_1 = get_molecule_mean_logits(smiles_1, tok2, mod2)
    print(shape_1[0][0].shape)
    print("Done with process 2")


def process(input_smiles):
    print("In process")
    tok, mod = get_chemBERT_tok_mod()
    shape_1 = get_molecule_mean_logits(input_smiles, tok, mod)
    print(shape_1[0][0].shape)
    print("Done with process")


tok, mod = get_chemBERT_tok_mod()

def smiles_to_inchikey(smiles):
    try:
        # Convert SMILES to RDKit Molecule
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError("Invalid SMILES string.")

        # Generate InChIKey
        inchi_key = inchi.MolToInchiKey(mol)
        return inchi_key
    except Exception as e:
        return str(e)


def get_dataset():
    with open("data/curated_chembl/chembl_data.json", "r") as f:
        # Save data_set as json
        data_set = json.load(f)["data"]

    return data_set


def get_args(dataset, fraq = 1.0):
    folder_count = 0
    folder_row_count = 0

    args = []

    if os.path.exists("data/curated_chembl/mol_embed_args.json"):
        with open("data/curated_chembl/mol_embed_args.json", "r") as f:
            args = json.load(f)["args"]
            return list(args)
    
    else:
        for row in tqdm(dataset, "Preparing args"):
            folder_row_count += 1
            smiles = row[0]
            inchi_key = smiles_to_inchikey(smiles)

            if inchi_key is None:
                continue
            
            file_name = f"data/curated_chembl/mol_embeds/{folder_count}/{inchi_key}_{folder_row_count}.h5"
            if not os.path.exists(f"data/curated_chembl/mol_embeds/{folder_count}"):
                    os.makedirs(f"data/curated_chembl/mol_embeds/{folder_count}")

            if folder_row_count % 100000 == 0:
                folder_count += 1
                folder_row_count = 0
            
            args.append((smiles, file_name, False))

            if len(args) > int(len(dataset) * fraq):
                break

        with open("data/curated_chembl/mol_embed_args.json", "w") as f:
            json.dump({"args": args}, f)

        return get_args(dataset, fraq=fraq)

    
