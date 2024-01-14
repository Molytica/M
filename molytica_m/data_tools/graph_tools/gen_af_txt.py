from molytica_m.data_tools import alpha_fold_tools
import os

af_uniprots = alpha_fold_tools.get_all_alphafold_uniprot_ids()

for species in af_uniprots.keys():
    print(species, len(af_uniprots[species]))
    af_uniprots_species = af_uniprots[species]

    if not os.path.exists(os.path.join("molytica_m", "data_tools", "protein_data", species)):
        os.makedirs(os.path.join("molytica_m", "data_tools", "protein_data", species))

    with open(os.path.join("molytica_m", "data_tools", "protein_data", species, "af_uniprots.txt"), "w") as file:
        for uniprot in af_uniprots_species:
            file.write(uniprot)
            file.write(" ")