import torch
import torch.nn as nn
import torch.nn.functional as F
from models.deepsets_invariant import DeepSetsInvariant

class DigitSumTextPhi(nn.Module):
    def __init__(self, embed_size: int):
        super().__init__()
        self.embed_size = embed_size

        self.embed = nn.Embedding(10, self.embed_size)
        #self.embed.weight.requires_grad=False
        self.fc1 = nn.Linear(embed_size, 50)
        self.dropout_fc1 = nn.Dropout()
        self.fc2 = nn.Linear(50, 10)
        self.dropout_fc2 = nn.Dropout()

    def forward(self, x):
        #x = x.squeeze(1)
        #x = self.embed(x)
        x = self.fc1(x)
        x = self.dropout_fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout_fc2(x)
        return x

class DigitSumTextRho(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(10, 10)
        self.fc2 = nn.Linear(10, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        return x

def digitsum_text50(embed_size=100):
    phi = DigitSumTextPhi(embed_size)
    rho = DigitSumTextRho()
    model = DeepSetsInvariant(phi, rho, embedding=True)
    return model

def digitsum_text(model_name):
    return{
        'digitsum_text50': digitsum_text50(),
    }[model_name]