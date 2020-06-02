import numpy as np
from models import digitsum_text
from models import deepsets_invariant
from torch import optim
import torch
import torch.nn as nn
import torch.functional as F
from torch import FloatTensor

### SETTINGS ###
NB_TRAIN_SETS = 100000
MAX_TRAIN_SET_SIZE = 10

NB_TEST_SETS = 100000
MIN_TEST_SET_SIZE = 5
MAX_TEST_SET_SIZE = 100

STEP_SIZE = 5

EMBED_SIZE = 100

### GENERATE TRAIN SETS ###

x_train = np.zeros((NB_TRAIN_SETS, MAX_TRAIN_SET_SIZE))
y_train = np.zeros((NB_TRAIN_SETS))

for i in range(NB_TRAIN_SETS):
    n = np.random.randint(1, MAX_TRAIN_SET_SIZE)
    for j in range(n):
        x_train[i,j] = np.random.randint(0,9)
    y_train[i] = np.sum(x_train[i])

x_train = torch.from_numpy(x_train).type(FloatTensor)
y_train = torch.from_numpy(y_train).type(FloatTensor).unsqueeze(1)

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

n_epochs = 20

model = digitsum_text(EMBED_SIZE)

if torch.cuda.is_available():
    model.cuda()

optimizer = optim.Adam(model.parameters(), lr= lr, weight_decay= wd)

loss_fn = nn.MSELoss(reduction = 'sum')

for epoch in range(n_epochs):
    out = model(x_train)
    loss = loss_fn(y_train, out)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()