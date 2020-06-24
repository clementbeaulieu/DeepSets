import torch
import torch.nn as nn
import torch.nn.functional as F
from models.deepsets_equivariant import DeepSetsEquivariant
from models.deepsets_equivariant import maxpool, sumpool

class CloudPointsRho(nn.Module):
    def __init__(self, dim, num_classes, sigma):
        super().__init__()
        self.dropout1 = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(dim, dim)
        self.sigma = sigma
        self.dropout2 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(dim, num_classes)

    def forward(self, x):
        x = self.dropout1(x)
        x = self.fc1(x)
        x = self.sigma(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return x

class CloudPoints(nn.Module):
    def __init__(self, dim_list, pool_list, sigma_list, out_sigma, num_classes):
        super().__init__()
        self.deepset = DeepSetsEquivariant(dim_list, pool_list, sigma_list)
        self.out_dim = dim_list[len(dim_list)-1]
        self.num_classes = num_classes
        self.out_sigma = out_sigma
        self.rho = CloudPointsRho(self.out_dim, self.num_classes, self.out_sigma)

    def forward(self, x):
        x = self.deepset(x)
        x = self.rho(x)
        return x

def cloudpoints_sumpool(deep_dim, num_classes, nb_layers=3):
    dim_list = [3]
    pool_list = []
    sigma_list = []
    for _ in range(nb_layers):
        dim_list.append(deep_dim)
        pool_list.append(sumpool)
        sigma_list.append(nn.ELU())
    model = CloudPoints(dim_list, pool_list, sigma_list, nn.ELU(), num_classes)
    return model

def cloudpoints_maxpool(deep_dim, num_classes, nb_layers=3):
    dim_list = [3]
    pool_list = []
    sigma_list = []
    for _ in range(nb_layers):
        dim_list.append(deep_dim)
        pool_list.append(sumpool)
        sigma_list.append(nn.Tanh())
    model = CloudPoints(dim_list, pool_list, sigma_list, nn.Tanh(), num_classes)
    return model

def cloudpoints(model_name, deep_dim, num_classes):
    return{
        'cloudpoints_sumpool': cloudpoints_sumpool(deep_dim, num_classes),
        'cloudpoints_maxpool': cloudpoints_maxpool(deep_dim, num_classes),
    }[model_name]