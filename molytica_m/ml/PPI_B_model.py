import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

class ProteinInteractionPredictor(nn.Module):
    def __init__(self, metadata_vector_size=18640, graph_feature_size=9):
        super(ProteinInteractionPredictor, self).__init__()
        
        # Metadata processing layers
        self.fc_a_1 = nn.Linear(metadata_vector_size, 128)
        self.fc_a_2 = nn.Linear(128, 128)

        #self.fc_b_1 = nn.Linear(metadata_vector_size, 128)
        #elf.fc_b_2 = nn.Linear(128, 128)

        # Graph processing layers (using GCN as an example)
        self.gcn_a_1 = GCNConv(graph_feature_size, 128)
        self.gcn_a_2 = GCNConv(128, 128)

        #self.gcn_b_1 = GCNConv(graph_feature_size, 128)
        #self.gcn_b_2 = GCNConv(128, 128)

        # Combining features
        self.fc_combined = nn.Linear(512, 128)  # Assuming combined features are concatenated

        # Final output layer
        self.output = nn.Linear(128, 1)

    def forward(self, metadata_a, metadata_b, x_a, edge_index_a, batch_vector_a, x_b, edge_index_b, batch_vector_b):
        # Process metadata
        metadata_a = F.relu(self.fc_a_1(metadata_a))
        metadata_a = F.relu(self.fc_a_2(metadata_a))

        metadata_b = F.relu(self.fc_a_1(metadata_b))
        metadata_b = F.relu(self.fc_a_2(metadata_b))

        # Process graph features
        x_a = F.relu(self.gcn_a_1(x_a, edge_index_a))
        x_a = F.relu(self.gcn_a_2(x_a, edge_index_a))
        x_a = global_mean_pool(x_a, batch_vector_a)  # Global average pooling

        x_b = F.relu(self.gcn_a_1(x_b, edge_index_b))
        x_b = F.relu(self.gcn_a_2(x_b, edge_index_b))
        x_b = global_mean_pool(x_b, batch_vector_b)  # Global average pooling

        # Combine features
        combined = torch.cat([metadata_a, metadata_b, x_a, x_b], dim=1)
        combined = F.relu(self.fc_combined(combined))

        # Output layer
        out = torch.sigmoid(self.output(combined))
        return out