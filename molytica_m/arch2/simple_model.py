import torch
import torch.nn as nn
import torch.nn.functional as F

class QSAR_model(nn.Module):
    def __init__(self, input_size):
        super(QSAR_model, self).__init__()
        self.fc1 = nn.Linear(input_size, 256+128)
        self.fc2 = nn.Linear(256+128, 256)
        self.fc3 = nn.Linear(256, 7)

    def forward(self, mol_embed, prot_embed):
        x = torch.cat((mol_embed, prot_embed), dim=1)
        x = F.relu(self.fc1(x))  # Apply ReLU after first layer
        x = F.relu(self.fc2(x))  # Apply ReLU after second layer
        x = self.fc3(x)
        return F.softmax(x, dim=1)  # Apply Softmax at the output layer
