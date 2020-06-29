import torch
import torch.nn as nn
import torch.nn.functional as F


def default_aggregator(x):
    out = torch.sum(x, dim=0, keepdim=True)
    return out


class DeepSetsInvariantBatch(nn.Module):

    def __init__(self, phi: nn.Module, rho: nn.Module, aggregator=default_aggregator, embedding=False):
        super().__init__()
        self.phi = phi
        self.rho = rho
        self.aggregator = aggregator
        self.embedding = embedding

    def forward(self, x, debug=False):
        '''
        Input x represents a set of size set_size with particles of dimensions particle_space_size.
        Input x of size (set_size, particle_space_size)
        '''

        if debug == True:
            print('x len: ', len(x))

        embed_list = []

        for input in x:

            if debug == True:
                print('input size: ', input.size())

            if self.embedding:
                input = input.squeeze(1)
                input = self.phi.embed(input)
            
            embed = self.phi(input)
            embed_list.append(embed)

        agg_list = []

        for embed in embed_list:
            # Aggregates the embeddings. By default, the aggregator is the usual sum.
            agg = self.aggregator(embed)

            if debug == True:
                print('agg size: ', agg.size())
            
            agg_list.append(agg)

        agg = torch.stack(agg_list, dim=0)
        agg = agg.squeeze(1)

        if debug == True:
            print('agg size: ', agg.size())

        # Process the invariant of the mebeddings through rho.
        out = self.rho(agg)

        if debug == True:
            print('out size: ', out.size())

        return out
