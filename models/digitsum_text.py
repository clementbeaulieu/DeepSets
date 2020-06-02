import torch
import torch.nn as nn
import torch.nn.functional as F
from models.deepsets_invariant import deepsets_invariant

class DigitSumPhi(nn.Module):
    def __init__(self, embed_size: int):
        super().__init__()
        self.embed_size = embed_size

        self.f1 = nn.Embedding(10, embed_size)
        self.f2 = nn.Linear(embed_size, 50)
        self.f3 = nn.ReLU()
        self.f4 = nn.Linear(50, 10)
        self.f5 = nn.ReLU()

    def forward(self, x):
        x = self.f1(x)
        x = self.f2(x)
        x = self.f3(x)
        x = self.f4(x)
        out = self.f5(x)
        return out

class DigitSumRho(nn.Module):
    def __init__(self, in_size: int, out_size: int):
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size

        self.f1 = nn.Linear(self.in_size, 10)
        self.f2 = nn.ReLU()
        self.f3 = nn.Linear(10, self.out_size)

    def forward(self, x):
        x = self.f1(x)
        x = self.f2(x)
        out = self.f3(x)
        return out

class DigitSumText(nn.Module):
    def __init__(self, embed_size):
        super().__init__()
        self.embed_size = embed_size

        self.phi = DigitSumPhi(embed_size)
        self.rho = DigitSumRho(10, 1)

        self.deepset = deepsets_invariant.DeepSetsInvariant(self.phi, self.rho)

    def forward(self, x):
        out = self.deepset.forward(x)
        return out
        
def digitsum_text(embed_size):
    model = DigitSumText(embed_size)
    return model

'''def digitsum_text(model_name, num_classes, input_channels, pretrained=False):
    return{
        'digitsum_text': digitsum_text(num_classes=num_classes, input_channels=input_channels),
    }[model_name]
    '''