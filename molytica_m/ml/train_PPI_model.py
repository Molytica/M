from molytica_m.data_tools.create_PPI_dataset import get_data_loader_and_size
from molytica_m.ml.PPI_model import ProteinInteractionPredictor
import torch.optim as optim
import torch.nn as nn
import torch

# Check if GPU is available and use it; otherwise, use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataloader, metadata_vector_size, graph_feature_size = get_data_loader_and_size()

# Initialize your model
model = ProteinInteractionPredictor(metadata_vector_size, graph_feature_size)

# Transfer the model to the device
model.to(device)

loss_function = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 1
for epoch in range(num_epochs):  # num_epochs is the number of epochs you want to train for
    for batch in dataloader:
        metadata_a, metadata_b, graph_data_a, graph_data_b, labels = batch
        metadata_a = metadata_a.to(device)
        metadata_b = metadata_b.to(device)
        graph_data_a = graph_data_a.to(device)
        graph_data_b = graph_data_b.to(device)
        labels = labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(metadata_a, metadata_b, graph_data_a, graph_data_b)

        # Compute loss
        loss = loss_function(outputs, labels)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        # Optional: print statistics
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')