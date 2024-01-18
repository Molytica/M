from molytica_m.data_tools.create_PPI_dataset import get_data_loader_and_size
from molytica_m.ml.PPI_model import ProteinInteractionPredictor
import torch.optim as optim
from tqdm import tqdm
import torch.nn as nn
import numpy as np
import torch
import sys

# Check if GPU is available and use it; otherwise, use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("cuda" if torch.cuda.is_available() else "cpu")

train_loader, val_loader, test_loader, metadata_vector_size, graph_feature_size = get_data_loader_and_size()

# Initialize your model
model = ProteinInteractionPredictor(metadata_vector_size, graph_feature_size)

# Transfer the model to the device
model.to(device)

loss_function = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop with progress bar
num_epochs = 100000  # Set the number of epochs
last_ten = []
val_max_acc = 0
for epoch in range(num_epochs):
    # Initialize tqdm progress bar
    with tqdm(train_loader, unit="batch") as tepoch:
        for batch in tepoch:
            tepoch.set_description(f"Epoch {epoch+1}")

            # Load batch data
            metadata_a, metadata_b, x_a, edge_index_a, x_b, edge_index_b, labels = batch
            metadata_a = metadata_a.to(device)
            metadata_b = metadata_b.to(device)
            x_a = x_a[0].to(device)
            edge_index_a = edge_index_a[0].to(device)
            x_b = x_b[0].to(device)
            edge_index_b = edge_index_b[0].to(device)
            labels = labels.to(device)

            # Forward and backward passes
            optimizer.zero_grad()
            outputs = model(metadata_a, metadata_b, x_a, edge_index_a, x_b, edge_index_b)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

            predicted = torch.round(outputs)
            correct = (predicted == labels).float().mean().item()

            # Update progress bar
            last_ten.append(correct)
            if len(last_ten) > 20000:
                last_ten = last_ten[-20000:]
            tepoch.set_postfix(loss=loss.item(), SMA_acc=np.mean(last_ten))

    val_correct = []

    with tqdm(val_loader, unit="batch") as tepoch:
        for batch in tepoch:
            tepoch.set_description(f"Validating epoch {epoch+1}")

            # Load batch data
            metadata_a, metadata_b, x_a, edge_index_a, x_b, edge_index_b, labels = batch
            metadata_a = metadata_a.to(device)
            metadata_b = metadata_b.to(device)
            x_a = x_a[0].to(device)
            edge_index_a = edge_index_a[0].to(device)
            x_b = x_b[0].to(device)
            edge_index_b = edge_index_b[0].to(device)
            labels = labels.to(device)

            # Forward and backward passes
            outputs = model(metadata_a, metadata_b, x_a, edge_index_a, x_b, edge_index_b)

            predicted = torch.round(outputs)
            correct = (predicted == labels).float().mean().item()

            # Update progress bar
            val_correct.append(correct)
            tepoch.set_postfix(Val_acc=np.mean(val_correct))

    val_acc = np.mean(val_correct)
    if val_acc > val_max_acc:
        print(f"Accuracy {val_acc} was better than previous best of {val_max_acc}. Saving model.")
        torch.save(model, 'molytica_m/ml/PPI_model_verify.pth')
        val_max_acc = val_acc

