from molytica_m.data_tools.create_PPI_dataset import get_data_loader_and_size
from molytica_m.ml.PPI_model import ProteinInteractionPredictor
import torch.optim as optim
from tqdm import tqdm
import torch.nn as nn
import torch

# Check if GPU is available and use it; otherwise, use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_loader, val_loader, test_loader, metadata_vector_size, graph_feature_size = get_data_loader_and_size()

# Initialize your model
model = ProteinInteractionPredictor(metadata_vector_size, graph_feature_size)

# Transfer the model to the device
model.to(device)

loss_function = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop with progress bar
num_epochs = 1  # Set the number of epochs
for epoch in range(num_epochs):
    # Initialize tqdm progress bar
    with tqdm(train_loader, unit="batch") as tepoch:
        for batch in tepoch:
            tepoch.set_description(f"Epoch {epoch+1}")

            # Load batch data
            metadata_a, metadata_b, graph_data_a, graph_data_b, labels = batch
            metadata_a = metadata_a.to(device)
            metadata_b = metadata_b.to(device)
            graph_data_a = graph_data_a.to(device)
            graph_data_b = graph_data_b.to(device)
            labels = labels.to(device)

            # Forward and backward passes
            optimizer.zero_grad()
            outputs = model(metadata_a, metadata_b, graph_data_a, graph_data_b)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

            # Update progress bar
            tepoch.set_postfix(loss=loss.item())