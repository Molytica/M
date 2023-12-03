from molytica_m.ml.PPI_model import ProteinInteractionPredictor
from molytica_m.data_tools.create_PPI_dataset import get_data_loader_and_size
import torch

# Check if GPU is available and use it; otherwise, use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataloader, metadata_vector_size, graph_feature_size = get_data_loader_and_size()

# Initialize your model
model = ProteinInteractionPredictor(metadata_vector_size, graph_feature_size)

# Transfer the model to the device
model.to(device)

# In your training loop, transfer data to the device
for batch in dataloader:
    metadata_a, metadata_b, graph_data_a, graph_data_b, labels = batch
    metadata_a = metadata_a.to(device)
    metadata_b = metadata_b.to(device)
    graph_data_a = graph_data_a.to(device)  # Make sure graph_data is compatible with device transfer
    graph_data_b = graph_data_b.to(device)  # Same as above
    labels = labels.to(device)

    # Forward pass
    outputs = model(metadata_a, metadata_b, graph_data_a, graph_data_b)
    # ... rest of your training loop ...
