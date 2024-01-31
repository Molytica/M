import json
import re

def split_string(input_string):
    pattern = r'(\d+)([a-zA-Z])(\d+)'
    match = re.match(pattern, input_string)

    if match:
        x = match.group(1)  # First group of digits
        letter = match.group(2)  # Letter
        y = match.group(3)  # Second group of digits
        return x, letter, y
    else:
        return "No match found"

def get_electron_configuration_vector(electron_configuration_string):
    max_electron_orbitals_string = "1s2 2s2 2p6 3s2 3p6 4s2 3d10 4p6 5s2 4d10 5p6 6s2 4f14 5d10 6p6 7s2 5f14 6d10 7p6 8s2"
    max_electron_orbitals = max_electron_orbitals_string.split(" ")
    electron_orbitals = electron_configuration_string.split(" ")
    order = ["1s", "2s", "2p", "3s", "3p", "4s", "3d", "4p", "5s", "4d", "5p", "6s", "4f", "5d", "6p", "7s", "5f", "6d", "7p", "8s"]

    max_dict = {}
    element_dict = {}
    valence_dict = {}
    
    for orbital in max_electron_orbitals:
        x, letter, y = split_string(orbital)
        max_dict[str(x) + letter] = int(y)
    
    for orbital in electron_orbitals:
        x, letter, y = split_string(orbital)
        element_dict[str(x) + letter] = int(y)

    for key in max_dict:
        if key not in element_dict:
            valence_dict[key] = max_dict[key]
        else:
            valence_dict[key] = max_dict[key] - element_dict[key]

    #print([valence_dict[key] for key in order])

    return [valence_dict[key] for key in order]


def main():
    element_datas = {}
    max_len = 0
    max_conf = ""
    vectors = {}

    with open("molytica_m/elements/PeriodicTableJSON.json", "r") as f:
        elements_json = json.load(f)
    
    for element in elements_json["elements"]:
        element_properties = {}

        element_properties["symbol"] = element["symbol"]
        element_properties["period"] = element["period"]
        element_properties["group"] = element["group"]
        element_properties["atomic_mass"] = element["atomic_mass"]
        element_properties["electron_affinity"] = element["electron_affinity"]
        element_properties["electronegativity_pauling"] = element["electronegativity_pauling"]
        element_properties["electron_configuration_vector"] = get_electron_configuration_vector(element["electron_configuration"])

        element_properties["total_vector_fingerprint"] = [element["period"], element["group"], element["atomic_mass"], element["electron_affinity"], element["electronegativity_pauling"]] + element_properties["electron_configuration_vector"]

        print(element_properties["total_vector_fingerprint"])

        vectors[element["symbol"]] = element_properties["total_vector_fingerprint"]

        if len(element["electron_configuration"]) > max_len:
            max_len = len(element["electron_configuration"])
            max_conf = element["electron_configuration"]

        element_datas[element["symbol"]] = element_properties

    #print(element_datas)
    #print(max_conf)
        
    # save the vector as json
    with open('molytica_m/elements/element_vectors.json', 'w') as fp:
        json.dump(vectors, fp)


if __name__ == "__main__":
    main()