from molytica_m.data_tools import create_PPI_dataset
from molytica_m.data_tools import alpha_fold_tools
from tqdm import tqdm
import torch
import h5py
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load("molytica_m/ml/PPI_S_model.pth", map_location=torch.device(device)).to(device)
model.eval()
model2 = torch.load("molytica_m/ml/PPI_C_model.pth", map_location=torch.device(device)).to(device)
model2.eval()

def predict_PPI_value(uniprot_A, uniprot_B):
    metadata_a = create_PPI_dataset.get_metadata(uniprot_A)
    metadata_b = create_PPI_dataset.get_metadata(uniprot_B)

    x_a, edge_index_a = create_PPI_dataset.get_graph(uniprot_A)
    x_b, edge_index_b = create_PPI_dataset.get_graph(uniprot_B)

    output = model(
        torch.tensor(metadata_a, dtype=torch.float).to(device).unsqueeze(0),
        torch.tensor(metadata_b, dtype=torch.float).to(device).unsqueeze(0),
        torch.tensor(x_a, dtype=torch.float).to(device),
        torch.tensor(edge_index_a, dtype=torch.long).to(device),
        torch.tensor(x_b, dtype=torch.float).to(device),
        torch.tensor(edge_index_b, dtype=torch.long).to(device),
    )

    PPI_value1 = float(output.to("cpu").detach().numpy()[0][0])

    output = model2(
        torch.tensor(metadata_a, dtype=torch.float).to(device).unsqueeze(0),
        torch.tensor(metadata_b, dtype=torch.float).to(device).unsqueeze(0),
        torch.tensor(x_a, dtype=torch.float).to(device),
        torch.tensor(edge_index_a, dtype=torch.long).to(device),
        torch.tensor(x_b, dtype=torch.float).to(device),
        torch.tensor(edge_index_b, dtype=torch.long).to(device),
    )

    PPI_value2 = float(output.to("cpu").detach().numpy()[0][0])

    return (PPI_value1 + PPI_value2) / 2

def get_PPI_value_single(uniprot_A, uniprot_B):
    save_path = 'data/PPI_interactome_probabilities/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    file_path = os.path.join(save_path, f"{uniprot_A}_{uniprot_B}.txt")
    
    if os.path.exists(file_path):
        with open(file_path, "r") as file:
            return float(file.read())
    else:
        PPI_value = predict_PPI_value(uniprot_A, uniprot_B)
        with open(file_path, "w") as file:
            file.write(str(PPI_value))
        return PPI_value
    
def get_PPI_value(A, B):
    return (get_PPI_value_single(A, B) + get_PPI_value_single(B, A)) / 2
    

if __name__ == "__main__":
    af_uniprots = alpha_fold_tools.get_alphafold_uniprot_ids()

    total_iterations = len(af_uniprots) ** 2
    progress_bar = tqdm(total=total_iterations, desc="Overall progress")

    for uniprot_A in af_uniprots:
        for uniprot_B in af_uniprots:
            get_PPI_value(uniprot_A, uniprot_B)
            progress_bar.update(1)

    progress_bar.close()