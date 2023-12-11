import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

class ProteinInteractionPredictor(nn.Module):
    def __init__(self, metadata_vector_size, graph_feature_size):
        super(ProteinInteractionPredictor, self).__init__()
        
        # Metadata processing layers
        self.fc1 = nn.Linear(metadata_vector_size, 256)
        self.fc2 = nn.Linear(256, 128)

        # Graph processing layers (using GCN as an example)
        self.gcn1 = GCNConv(graph_feature_size, 128)
        self.gcn2 = GCNConv(128, 128)

        # Combining features
        self.fc_combined = nn.Linear(512, 256)  # Assuming combined features are concatenated
        self.fc_combined2 = nn.Linear(256, 128)

        # Final output layer
        self.output = nn.Linear(128, 1)

    def forward(self, metadata_a, metadata_b, x_a, edge_index_a, x_b, edge_index_b):
        # Process metadata
        metadata_a = F.relu(self.fc1(metadata_a))
        metadata_a = F.relu(self.fc2(metadata_a))

        metadata_b = F.relu(self.fc1(metadata_b))
        metadata_b = F.relu(self.fc2(metadata_b))

        # Process graph features
        x_a = F.relu(self.gcn1(x_a, edge_index_a))
        x_a = F.relu(self.gcn2(x_a, edge_index_a))
        x_a = F.relu(self.gcn2(x_a, edge_index_a))
        x_a = global_mean_pool(x_a, torch.zeros(x_a.size(0), dtype=torch.long, device=x_a.device))  # Global average pooling

        x_b = F.relu(self.gcn1(x_b, edge_index_b))
        x_b = F.relu(self.gcn2(x_b, edge_index_b))
        x_b = F.relu(self.gcn2(x_b, edge_index_b))
        x_b = global_mean_pool(x_b, torch.zeros(x_b.size(0), dtype=torch.long, device=x_b.device))  # Global average pooling

        # Combine features
        combined = torch.cat([metadata_a, metadata_b, x_a, x_b], dim=1)
        combined = F.relu(self.fc_combined(combined))
        combined = F.relu(self.fc_combined2(combined))

        # Output layer
        out = torch.sigmoid(self.output(combined))
        return out