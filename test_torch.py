import torch
import numpy as np

def digitsum_score(output, target):
    output = output.data
    target = target.data
    torch.round(output)
    torch.round(target)

    '''
    output = output.t()
    correct = output.eq(target.view(1, -1).expand_as(output))

    correct_k = correct[:1].view(-1).float().sum(0)
    correct_k.mul_(1.0 / batch_size)

    res = correct_k.clone()

    return res.item(), output, target
    '''
    batch_size = target.size(0)
    acc = torch.sum(output == target, dtype=torch.double)
    return acc.item()/batch_size

x = torch.zeros((2,1), requires_grad=True)
x[0][0] = 1.5
x[1][0] = 2.8

y = torch.zeros((2,1), requires_grad=True)
y[0][0] = 1.5
y[1][0] = 1.1

print(digitsum_score(x, y))