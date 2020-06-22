import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class DigitSumTextLoader(Dataset):

    def __init__(self, data_dir, split, min_size, max_size, dataset_size, train = True, custom_transforms = None):
        self.data_dir = data_dir
        self.split = split
        self.min_size = min_size
        self.max_size = max_size
        self.dataset_size = dataset_size
        self.train = train

        digits = np.arange(0, 10)

        # Sampling the sets sizes randomly given min_size and max_size.
        set_sizes_range = np.arange(self.min_size, self.max_size + 1)
        set_sizes = np.random.choice(set_sizes_range, self.dataset_size, replace = True)
        self.digit_items = []

        # Adding randomly sampled particles (text digits) into sets of the dataset.
        for i in range(self.dataset_size):
            set_items = torch.from_numpy(np.random.choice(digits, set_sizes[i], replace = True)).type(torch.LongTensor).unsqueeze(1)
            self.digit_items.append(set_items)

        self.data = [self.__getitem__(item) for item in range(self.__len__())]

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, item):
        set_items = self.digit_items[item]

        out = np.array([[0.0]])
        digit_list = []

        for digit_item in set_items:
            out[0][0] += digit_item
            digit_list.append(digit_item)

        out = torch.from_numpy(out).type(torch.LongTensor)

        return torch.stack(digit_list, dim = 0).type(torch.LongTensor), out
    
    def get_label(self, item):
        return self.__getitem__(item)[1]

def compute_batch(batch, model):
    target_list = []
    output_list = []
    #batch_size=len(batch)
    #target_size=batch[0][1].size()[0]
    for (input, target) in batch:
        #input, target = input.to(args.device), target.to(args.device)
        target_list.append(target)

        if model.embedding:
            input = input.squeeze(1)
            input = model.phi.embed(input)
        
        input = input.requires_grad_()
        output = model(input)
        output_list.append(output)
    output_batch = torch.stack(output_list, dim=0)
    target_batch = torch.stack(target_list, dim=0)
    output_batch = output_batch.squeeze(1)
    target_batch = target_batch.squeeze(1).type(torch.FloatTensor)
    return output_batch, target_batch

data_dir = 'Users/clementbeaulieu/'
dataloader = torch.utils.data.DataLoader(DigitSumTextLoader(data_dir, 'train', 2, 5, 1), batch_size=1, shuffle=True, num_workers=4, collate_fn=lambda x: x, pin_memory=True)

import torch.nn as nn
import torch.nn.functional as F
embed_size = 25


class DigitSumTextPhi(nn.Module):
    def __init__(self, embed_size: int):
        super().__init__()
        self.embed_size = embed_size

        self.embed = nn.Embedding(10, self.embed_size)
        self.fc1 = nn.Linear(self.embed_size, 50)
        self.dropout_fc1 = nn.Dropout()
        self.fc2 = nn.Linear(50, 10)
        self.dropout_fc2 = nn.Dropout()

    def forward(self, x):
        #x = x.squeeze(1)
        #x = self.embed(x)
        #x = x.type(torch.FloatTensor)
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

def default_aggregator(x):
    out = torch.sum(x, dim=0, keepdim=True)
    return out


class DeepSetsInvariant(nn.Module):

    def __init__(self, phi: nn.Module, rho: nn.Module, aggregator=default_aggregator, embedding=False, debug=False):
        super().__init__()
        self.phi = phi
        self.rho = rho
        self.aggregator = aggregator
        self.embedding = embedding
        self.debug = debug

    def forward(self, x):
        '''
        Input x represents a set of size set_size with particles of dimensions particle_space_size.
        Input x of size (set_size, particle_space_size)
        '''

        if self.debug == True:
            print('x size: ', x)
            print('x: ', x)
        # Computes the embeddings of each particle in population x.
        embed = self.phi(x)

        if self.debug == True:
            print('embed size: ', embed.size())
            print('embed: ', embed)

        # Aggregates the embeddings. By default, the aggregator is the usual sum.
        agg = self.aggregator(embed)

        if self.debug == True:
            print('agg size: ', agg.size())
            print('agg: ', agg)

        # Process the invariant of the mebeddings through rho.
        out = self.rho(agg)

        if self.debug == True:
            print('out size: ', out.size())
            print('out: ', out)

        return out

def digitsum_text50(embed_size=100):
    phi = DigitSumTextPhi(embed_size)
    rho = DigitSumTextRho()
    model = DeepSetsInvariant(phi, rho, embedding = True, debug=True)
    return model

model = digitsum_text50(embed_size)

for batch in dataloader:
    output_batch, target_batch = compute_batch(batch, model)

    print(output_batch, target_batch)

    criterion = nn.MSELoss(reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    loss = criterion(output_batch, target_batch)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()