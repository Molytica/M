from molytica_m.chembl_curation import get_chembl_data


def get_input_from_smiles_target_and_modulation(smiles, target_uniprot, modulation):
    
    # SMILES, molecule descriptors, molecule graph, protein metadata and protein graph are loaded directly from h5 files preexistant
    molecule_descriptors = get_chembl_data.load_molecule_descriptors(smiles)
    molecule_graph = get_chembl_data.load_molecule_graph(smiles)
    protein_metadata = get_chembl_data.load_protein_metadata(target_uniprot)
    protein_graph = get_chembl_data.load_protein_graph(target_uniprot)

    # These are generated on the fly and then saved as h5 files for future use
    smiles_embedding = get_chembl_data.load_smiles_embedding(smiles)
    protein_embedding = get_chembl_data.load_protein_embedding(target_uniprot)

    return smiles, molecule_descriptors, molecule_graph, protein_metadata, protein_graph, smiles_embedding, protein_embedding


def train_on_data_early_stop(train_data, val_data):
    pass

def test_on_data(test_data):
    pass


def train():
    CV_split = get_chembl_data.get_categorised_CV_split()
    test_metricss = []

    for k in range(len(CV_split)):
        train_data = []
        test_data = []
        val_data = []
        for i in range(len(CV_split)):
            if i == k:
                split_point = len(CV_split) // 2
                test_data += CV_split[:split_point]
                val_data += CV_split[split_point:] 
            else:
                train_data += CV_split[i]

        print("Train data:", len(train_data))
        print("Test data:", len(test_data))
        print("Val data:", len(val_data))

        # Train the model
        train_on_data_early_stop(train_data, val_data)

        # Test the model
        test_metrics = test_on_data(test_data)
        test_metricss.append(test_metrics)


def main():
    train()

if __name__ == "__main__":
    main()
