import torch
import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(44000, 22000),
            nn.ReLU(),
            nn.Linear(22000, 11000),
            nn.ReLU(),
            nn.Linear(11000, 2200),
            nn.ReLU(),
            nn.Linear(2200, 220)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(220, 2200),
            nn.ReLU(),
            nn.Linear(2200, 11000),
            nn.ReLU(),
            nn.Linear(11000, 22000),
            nn.ReLU(),
            nn.Linear(22000, 44000)
        )
        
        self.dropout = nn.Dropout(p=0.2)
        self.attention = nn.MultiheadAttention(embed_dim=220, num_heads=2)
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.dropout(x)
        x, _ = self.attention(x, x, x)
        x = self.decoder(x)
        return x



def get_conversion_model():
    model = Autoencoder()
    return model
