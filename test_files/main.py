import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
#from torch import FloatTensor
import time
import sys

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
#print(device)

def default_aggregator(x: torch.FloatTensor, keep_dim = True):
    return torch.sum(x, dim = 0, keep_dim = keep_dim)

class DeepSetsInvariant(nn.Module):

    def __init__(self, phi: nn.Module, rho: nn.Module, aggregator = default_aggregator):
        super().__init__()
        self.phi = phi
        self.rho = rho
        self.aggregator = aggregator

    def forward(self, x):
        # Computes the embeddings of each particle in population x.
        embed = self.phi.forward(x)
        # Aggregates the embeddings. By default, the aggregator is the usual sum.
        agg = torch.sum(embed, dim = 0)
        # Process the invariant of the mebeddings through rho.
        out = self.rho.forward(agg)
        return out

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
        x = x.type(torch.FloatTensor)
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

        self.deepset = DeepSetsInvariant(self.phi, self.rho)

    def forward(self, x):
        out = self.deepset(x)
        return out
        
def digitsum_text(embed_size):
    model = DigitSumText(embed_size)
    return model

from torch import optim

### SETTINGS ###
NB_TRAIN_SETS = 10000
MAX_TRAIN_SET_SIZE = 10

NB_TEST_SETS = 10000
MIN_TEST_SET_SIZE = 5
MAX_TEST_SET_SIZE = 100

STEP_SIZE = 5

EMBED_SIZE = 100

### GENERATE TRAIN SETS ###

x_train = np.zeros((NB_TRAIN_SETS, MAX_TRAIN_SET_SIZE))
y_train = np.zeros((NB_TRAIN_SETS, 1))



for i in range(NB_TRAIN_SETS):
    n = np.random.randint(1, MAX_TRAIN_SET_SIZE)
    for j in range(n):
        x_train[i,j] = np.random.randint(0,9)
    y_train[i] = np.sum(x_train[i])

dtype = torch.FloatTensor

x_train_tensor = torch.LongTensor(x_train)
y_train_tensor = torch.LongTensor(y_train).type(dtype).unsqueeze(1)

### CREATION OF THE DATALOADER ASSOCIATED to x_train and y_train

from torch.utils import data

dataset_train = data.TensorDataset(x_train_tensor, y_train_tensor)
dataloader_train = data.DataLoader(dataset_train, batch_size=5, shuffle=True)

#print(dataset_train[0])

### GENERATE TEST SETS ###

def gen_test_data(num_sets, length):
    x_test = np.zeros((num_sets, length))
    y_test = np.zeros((num_sets))
    for i in range(num_sets):
        for j in range(length):
            x_test[i,j] = np.random.randint(0,9)
        y_test[i] = np.sum(x_test[i])
    return x_test, y_test

### MODEL ###
lr = 1e-3
wd = 5e-3

model = digitsum_text(EMBED_SIZE)

optimizer = optim.Adam(model.parameters(), lr = lr, weight_decay = wd)

loss_fn = nn.MSELoss(reduction = 'sum')

if torch.cuda.is_available():
        model.cuda()

def train(train_loader, model, criterion, optimizer, epoch, eval_score = None, print_freq = 10, lr = 1e-3, wd = 5e-3):

    '''# switch to train mode
    model.train()
    meters = {}
    meters['logger'] = 'train'
    meters['acc1'] = 0.0
    meters['batch_time'] = 0.0
    end = time.time()'''

    for i , (input, target) in enumerate(train_loader):
        '''# print(f'{i} - {input.size()} - {target_class.size()}')
        batch_size = input.size(0)
        '''

        #print(input.size())

        #sys.exit()

        input, target = input.to(device), target.to(device)

        output = model(input)

        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        '''if eval_score is not None:
            acc1, pred, label = eval_score(output, target)
            meters['acc1'].update(acc1, n=batch_size)'''


        '''# measure elapsed time
        meters['batch_time'].update(time.time() - end, n=batch_size)
        end = time.time()'''

        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {3}'.format(
                   epoch, i, len(train_loader), loss))

        '''# measure elapsed time
        meters['batch_time'].update(time.time() - end, n=batch_size)
        end = time.time()'''

for epoch in range(20):
    train(dataloader_train, model, loss_fn, optimizer, epoch, lr, wd)