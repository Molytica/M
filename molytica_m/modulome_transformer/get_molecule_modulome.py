from concurrent.futures import ProcessPoolExecutor
from molytica_m.arch_2 import modulation_model

modulation_model = modulation_model.get_trained_model()

def get_data(SMILES, uniprot_id):
    data = ("", [], 0)
    return data


def expand_id_pair_to_data(arg):
    SMILES, uniprot_id = arg

    input_data = get_data(SMILES, uniprot_id)

    return input_data


def get_modulome(SMILES):
    # Your code here

    uniprot_list = []

    id_input_tuples = []

    for uniprot_id in uniprot_list:
        id_input_tuples.append((SMILES, uniprot_id))

    with ProcessPoolExecutor() as executor:
        data_input_tuples = executor.map(expand_id_pair_to_data, id_input_tuples)


    modulome = modulation_model.predict(data_input_tuples)
    
    return modulome

def main():
    # Call create_modulome function with sample inputs
    SMILES = "C1=CC=CC=C1"
    result = get_modulome(SMILES)
    print(result)

if __name__ == "__main__":
    main()
