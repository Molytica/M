# Load model directly
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch
import numpy as np

def get_chemBERT_tok_mod():
    tokenizer = AutoTokenizer.from_pretrained("DeepChem/ChemBERTa-77M-MLM")
    model = AutoModelForMaskedLM.from_pretrained("DeepChem/ChemBERTa-77M-MLM")

    return tokenizer, model

def get_molecule_mean_logits(smiles, tokenizer, model):
    # Tokenize the input SMILES string
    with torch.no_grad():
        inputs = tokenizer(smiles, return_tensors="pt")

        # Perform inference with the model
        outputs = model(**inputs).logits.mean(dim=1, keepdim=True)

        return outputs

if __name__ == "__main__":

    tok, mod = get_chemBERT_tok_mod()
    # Test the model
    smiles_1 = "CC(=O)OC1=CC=CC=C1C(=O)O"
    shape_1 = get_molecule_mean_logits(smiles_1, tok, mod)
    print(shape_1[0][0].shape)

    smiles_1 = "CC(=O)OC1==O)OC1=CC=CC=CC=CC=C1C(=O)O"
    shape_1 = get_molecule_mean_logits(smiles_1, tok, mod)
    print(shape_1[0][0].shape)

