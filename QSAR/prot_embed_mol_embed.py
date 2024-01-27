import molytica_m.chembl_curation.get_chembl_data as get_chembl_data

def load_protein_embed(uniprot_id): # 1024 dim vector
    get_chembl_data.load_protein_embedding(uniprot_id)

def load_molecule_embed(smiles): # 600 dim vector
    get_chembl_data.load_molecule_embedding(smiles)

def get_CV_dataset():
    get_chembl_data.get_categorised_data()


if __name__ == "__main__":
    print(load_molecule_embed(""))