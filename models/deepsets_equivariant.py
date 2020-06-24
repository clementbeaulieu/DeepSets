import torch
import torch.nn as nn
import torch.nn.functional as F

def maxpool(x):
    res, _ = torch.max(x, dim=0, keepdim=True)
    return res

def sumpool(x):
    res = torch.sum(x, dim=0, keepdim=True)
    return res

class EquivariantLayer(nn.Module):
    def __init__(self, in_channels, out_channels, agg=maxpool, sigma=nn.Tanh()):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.agg = agg
        self.fc = nn.Linear(self.in_channels, self.out_channels)
        self.sigma = sigma

    def forward(self, x):
        # Aggregate particles.
        agg = self.agg(x)

        # Computes difference between x and the product of aggregates/row tensor with ones.
        M = x.size(0)
        one_tensor = torch.ones(M,1)
        x = x - torch.matmul(one_tensor, agg)

        x = self.fc(x)
        x = self.sigma(x)

        return x

class DeepSetsEquivariant(nn.Module):

    def __init__(self, dim_list, pool_list, sigma_list, embedding=False):
        super().__init__()

        assert(len(dim_list) > 1)
        assert(len(dim_list) - 1 == len(pool_list))
        assert(len(dim_list) - 1 == len(sigma_list))

        self.dim_list = dim_list
        self.pool_list = pool_list
        self.sigma_list = sigma_list
        self.nb_layers = len(dim_list)-1

        self.layers = []
        self.add_layers()

        self.embedding = embedding

    def add_layer(self, in_dim, out_dim, pool, sigma):
        layer = EquivariantLayer(in_dim, out_dim, pool, sigma)
        self.layers.append(layer)

    def add_layers(self):
        for i in range(self.nb_layers):
            in_dim = self.dim_list[i]
            out_dim = self.dim_list[i+1]
            pool = self.pool_list[i]
            sigma = self.sigma_list[i]
            self.add_layer(in_dim, out_dim, pool, sigma)

    def forward(self, x, debug=False):
        for i in range(self.nb_layers):
            x = self.layers[i](x)
        return x