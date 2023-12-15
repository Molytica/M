import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

class ProteinModulationPredictor(nn.Module):
    def __init__(self, metadata_vector_size=18640, mol_properties_vector_size=21, graph_feature_size=9):
        super(ProteinModulationPredictor, self).__init__()
        
        # Metadata processing layers
        self.fc1 = nn.Linear(metadata_vector_size, 128)
        self.fc2 = nn.Linear(128, 128)

        self.fcm1 = nn.Linear(mol_properties_vector_size, 64)
        self.fcm2 = nn.Linear(64, 64)

        # Graph processing layers (using GCN as an example)
        self.gcn_prot_1 = GCNConv(graph_feature_size, 128)
        self.gcn_prot_2 = GCNConv(128, 128)

        self.gcn_mol_1 = GCNConv(graph_feature_size, 128)
        self.gcn_mol_2 = GCNConv(128, 128)

        # Combining features
        self.fc_combined = nn.Linear(128 * 2 + 128 + 64, 128)  # Assuming combined features are concatenated

        # Final output layer
        self.output = nn.Linear(128, 2) # Will output IC50 and EC50 value respectively

    def forward(self, metadata_prot, metadata_mol, x_prot, edge_index_prot, batch_vector_prot, x_mol, edge_index_mol, batch_vector_mol):
        # Process metadata
        metadata_prot = F.relu(self.fc1(metadata_prot))
        metadata_prot = F.relu(self.fc2(metadata_prot))

        metadata_mol = F.relu(self.fcm1(metadata_mol))
        metadata_mol = F.relu(self.fcm2(metadata_mol))

        # Process graph features
        x_prot = F.relu(self.gcn_prot_1(x_prot, edge_index_prot))
        x_prot = F.relu(self.gcn_prot_2(x_prot, edge_index_prot))
        x_prot = global_mean_pool(x_prot, batch_vector_prot)  # Global average pooling

        x_mol = F.relu(self.gcn_mol_1(x_mol, edge_index_mol))
        x_mol = F.relu(self.gcn_mol_2(x_mol, edge_index_mol))
        x_mol = global_mean_pool(x_mol, batch_vector_mol)  # Global average pooling

        # Combine features
        combined = torch.cat([metadata_prot, metadata_mol, x_prot, x_mol], dim=1)
        combined = F.relu(self.fc_combined(combined))

        # Output layer
        out = F.relu(self.output(combined))
        return out