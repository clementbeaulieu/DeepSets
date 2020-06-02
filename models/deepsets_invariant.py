import torch
import torch.nn as nn
import torch.nn.functional as F


def default_aggregator(x):
    out = torch.sum(x, dim=0, keepdim=True)
    return out


class DeepSetsInvariant(nn.Module):

    def __init__(self, phi: nn.Module, rho: nn.Module, aggregator=default_aggregator):
        super().__init__()
        self.phi = phi
        self.rho = rho
        self.aggregator = aggregator

    def forward(self, x, debug=False):
        '''
        Input x represents a set of size set_size with particles of dimensions particle_space_size.
        Input x of size (set_size, particle_space_size)
        '''

        # Computes the embeddings of each particle in population x.
        embed = self.phi(x)

        # Aggregates the embeddings. By default, the aggregator is the usual sum.
        agg = self.aggregator(embed)

        # Process the invariant of the mebeddings through rho.
        out = self.rho(agg)

        if debug == True:
            print('x size: ', x.size())
            print('embed size: ', embed.size())
            print('agg size: ', agg.size())
            print('out size: ', out.size())

        return out
