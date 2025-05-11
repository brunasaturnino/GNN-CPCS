import torch.nn as nn
import torch

class GCN(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 2 * out_features)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(2 * out_features, out_features)

    def forward(self, x, adj):
        h = torch.matmul(adj, x)
        h = self.fc1(h)
        h = self.relu(h)
        h = self.fc2(h)
        return h

class GCN_Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.encoder = GCN(input_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, input_dim)

    def forward(self, x, adj):
        z = self.encoder(x, adj)
        x_hat = self.decoder(z)
        return x_hat, z
    
