from tqdm import tqdm
import random
from molytica_m.data_tools import alpha_fold_tools
from molytica_m.data_tools import create_PPI_dataset
import numpy as np
from molytica_m.data_tools.graph_tools import interactome_tools
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data, Batch
from molytica_m.ml.PPI_B_model import ProteinInteractionPredictor
import sys
import json

with open("molytica_m/data_tools/filtered_no_reverse_duplicates_huri_and_biogrid_af_uniprot_edges.json", "r") as file:
    edge_list = json.load(file)["filtered_no_reverse_duplicates_huri_and_biogrid_af_uniprot_edges"]

def create_batch(uniprot_pairs_batch):
    metadata_as = []
    metadata_bs = []

    data_as = []
    data_bs = []

    for uniprot_pair in uniprot_pairs_batch:
        uniprot_A = uniprot_pair[0]
        uniprot_B = uniprot_pair[1]

        metadata_as.append(create_PPI_dataset.get_metadata(uniprot_A))
        metadata_bs.append(create_PPI_dataset.get_metadata(uniprot_B))
        
        x_a, edge_index_a = create_PPI_dataset.get_graph_raw(uniprot_A)
        x_b, edge_index_b = create_PPI_dataset.get_graph_raw(uniprot_B)

        data_as.append(Data(x=torch.Tensor(x_a), edge_index=torch.tensor(edge_index_a, dtype=torch.long)))
        data_bs.append(Data(x=torch.Tensor(x_b), edge_index=torch.tensor(edge_index_b, dtype=torch.long)))

    metadata_as = np.array(metadata_as)
    metadata_bs = np.array(metadata_bs)
    metadata_as = torch.tensor(metadata_as, dtype=torch.float).to(device)
    metadata_bs = torch.tensor(metadata_bs, dtype=torch.float).to(device)

    batch_as = Batch.from_data_list(data_as).to(device)
    batch_bs = Batch.from_data_list(data_bs).to(device)
    return metadata_as, metadata_bs, batch_as, batch_bs

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ProteinInteractionPredictor().to(device)
loss_function = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

combined_edge_list = interactome_tools.get_full_edge_list()
af_uniprots = alpha_fold_tools.get_alphafold_uniprot_ids()
batch_size = 30

def count_matches(outputs, labels):
    assert outputs.shape == labels.shape, "not same size"

    # Convert outputs to the same type as labels (usually integer) after rounding
    rounded_outputs = outputs.round().type_as(labels)

    # Ensure that both tensors are of the same shape
    if rounded_outputs.shape != labels.shape:
        # Adjust shapes if necessary; use .squeeze() or .view() based on your specific needs
        rounded_outputs = rounded_outputs.squeeze()
        labels = labels.squeeze()

    # Calculate matches
    matches = (rounded_outputs == labels)

    # Count the number of True values
    num_matches = matches.sum().item()

    return num_matches

random.shuffle(combined_edge_list)
l_comb_edge_list = len(combined_edge_list)

train_split = int(l_comb_edge_list * 0.8)
val_split = int(l_comb_edge_list * 0.9)

train_edge_list = combined_edge_list[:train_split]
val_edge_list = combined_edge_list[train_split:val_split]
test_edge_list = combined_edge_list[val_split:]

l_train_edge_list = len(train_edge_list)
epoch = 0
best_val_acc = 0
acc_train = []

while True:
    epoch += 1

    model.train()
    var_lab = None
    total_acc = 0
    total_elements = 0
    accs = []
    with tqdm(total=len(train_edge_list), desc=f"Epoch {epoch} Training") as pbar:
        for idx in range(0, len(train_edge_list), batch_size):
            batch = []
            labels = []

            real_batch_size = 0
            for x in range(min(batch_size, l_train_edge_list - idx)):
                batch.append(combined_edge_list[idx + x])
                labels.append(1)

                # Add negative example
                uniprot_A, uniprot_B = random.choice(af_uniprots), random.choice(af_uniprots)
                while [uniprot_A, uniprot_B] in edge_list or [uniprot_B, uniprot_A] in edge_list:
                    uniprot_A, uniprot_B = random.choice(af_uniprots), random.choice(af_uniprots)
                    
                batch.append([random.choice(af_uniprots), random.choice(af_uniprots)])
                labels.append(0)

                real_batch_size += 2

            metadata_as, metadata_bs, batch_as, batch_bs = create_batch(batch)

            outputs = model(metadata_as, metadata_bs, batch_as.x, batch_as.edge_index, batch_as.batch, batch_bs.x, batch_bs.edge_index, batch_bs.batch)
            labels = torch.tensor(np.array(labels), dtype=torch.float).unsqueeze(1).to(device)
            acc = count_matches(outputs, labels) / outputs.shape[0]
            accs.append(acc)
            if len(accs) > 1000:
                accs = accs[-1000:]
            sma_acc = np.mean(accs)
            total_acc += acc
            total_elements += 1
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            var = torch.var(outputs.squeeze())
            if not var_lab:
                var_lab = torch.var(labels.squeeze())

            pbar.set_description(f"Epoch {epoch} Training - Loss: {loss:.4f}, Acc: {acc:.4f},  Avg Acc: {total_acc/total_elements:.4f}, SMA Acc: {sma_acc:.4f}, Var: {var:.4f}, Var Lab: {var_lab:.4f}")
            pbar.update(int(real_batch_size/2))
    

    model.eval()
    l_val_edge_list = len(val_edge_list)
    total_acc = 0
    total_elements = 0
    accs = []
    with tqdm(total=len(val_edge_list), desc=f"Epoch {epoch} Validation") as pbar:
        for idx in range(0, len(val_edge_list), batch_size):
            batch = []
            labels = []

            real_batch_size = 0
            for x in range(min(batch_size, l_val_edge_list - idx)):
                batch.append(combined_edge_list[idx + x])
                labels.append(1)

                # Add negative example
                batch.append([random.choice(af_uniprots), random.choice(af_uniprots)])
                labels.append(0)

                real_batch_size += 2

            metadata_as, metadata_bs, batch_as, batch_bs = create_batch(batch)

            outputs = model(metadata_as, metadata_bs, batch_as.x, batch_as.edge_index, batch_as.batch, batch_bs.x, batch_bs.edge_index, batch_bs.batch)
            labels = torch.tensor(np.array(labels), dtype=torch.float).unsqueeze(1).to(device)
            acc = count_matches(outputs, labels) / outputs.shape[0]
            accs.append(acc)
            if len(accs) > 1000:
                accs = accs[-1000:]
            sma_acc = np.mean(accs)
            total_acc += acc
            total_elements += 1
            loss = loss_function(outputs, labels)
            var = torch.var(outputs.squeeze())
            if not var_lab:
                var_lab = torch.var(labels.squeeze())

            pbar.set_description(f"Epoch {epoch} Training - Loss: {loss:.4f}, Acc: {acc:.4f},  Avg Acc: {total_acc/total_elements:.4f}, SMA Acc: {sma_acc:.4f}, Var: {var:.4f}, Var Lab: {var_lab:.4f}")
            pbar.update(int(real_batch_size/2))
    
    val_acc = total_acc / total_elements
    if val_acc > best_val_acc:
        print(f"New val_acc {val_acc} was better than previous of {best_val_acc}. Saving new model.")
        torch.save(model, "molytica_m/ml/PPI_B_model.pth")
        best_val_acc = val_acc
