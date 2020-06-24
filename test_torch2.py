import torch

def maxpool(x):
    M = x.size(0)
    
    res, _ = torch.max(x, dim=0, keepdim=True)
    return res

x = torch.zeros((4, 10))
x[0][6] = 4.0
x[1][6] = 8.0
x[0][8] = 5.0
x[3][8] = 2.0

M = x.size(0)
one_tensor = torch.ones(M,1)
y = maxpool(x)
ym = torch.matmul(one_tensor, y)

print(y)
print(ym)

def var(x):
    res, _ = x.max(1, keepdim=True)
    return res

print(var(x))
print(x-var(x))