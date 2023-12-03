from molytica_m.data_tools import alpha_fold_tools

af_uniprots = alpha_fold_tools.get_alphafold_uniprot_ids()

with open("molytica_m/data_tools/af_uniprots_txt.txt", "w") as file:
    for uniprot in af_uniprots:
        file.write(uniprot)
        file.write(" ")