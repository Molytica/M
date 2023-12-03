import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class ProteinInteractionPredictor(nn.Module):
    def __init__(self, metadata_vector_size, graph_feature_size):
        super(ProteinInteractionPredictor, self).__init__()
        
        # Metadata processing layers
        self.fc1 = nn.Linear(metadata_vector_size, 128)
        self.fc2 = nn.Linear(128, 128)

        # Graph processing layers (using GCN as an example)
        self.gcn1 = GCNConv(graph_feature_size, 128)
        self.gcn2 = GCNConv(128, 128)

        # Combining features
        self.fc_combined = nn.Linear(512, 128)  # Assuming combined features are concatenated

        # Final output layer
        self.output = nn.Linear(128, 1)

    def forward(self, metadata_a, metadata_b, graph_data_a, graph_data_b):
        # Process metadata
        metadata_a = F.relu(self.fc1(metadata_a))
        metadata_a = F.relu(self.fc2(metadata_a))

        metadata_b = F.relu(self.fc1(metadata_b))
        metadata_b = F.relu(self.fc2(metadata_b))

        # Process graph data
        x_a, edge_index_a = graph_data_a.x, graph_data_a.edge_index
        x_b, edge_index_b = graph_data_b.x, graph_data_b.edge_index

        x_a = F.relu(self.gcn1(x_a, edge_index_a))
        x_a = F.relu(self.gcn2(x_a, edge_index_a))

        x_b = F.relu(self.gcn1(x_b, edge_index_b))
        x_b = F.relu(self.gcn2(x_b, edge_index_b))

        # Combine features
        combined = torch.cat([metadata_a, metadata_b, x_a, x_b], dim=1)
        combined = F.relu(self.fc_combined(combined))

        # Output layer
        out = torch.sigmoid(self.output(combined))
        return out