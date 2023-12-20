import torch
from molytica_m.data_tools.graph_tools import graph_tools
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = torch.load("molytica_m/ml/iP_B_model.pth").to(device)

mol_features, mol_csr_matrix = graph_tools.get_raw_graph_from_smiles_string(smiles)