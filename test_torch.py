import torch
import numpy as np

x = np.zeros((2,3), dtype=np.int64)
x[0][0] = 1
x[1][0] = 2

y = np.zeros((2,3), dtype=np.int64)
y[0][0] = 0
y[1][0] = 2

def fast_hist(pred, label, n):
    k = (label >= 0) & (label < n)
    return np.bincount(
        n * label[k].astype(int) + pred[k], minlength=n ** 2).reshape(n, n)

k = (y>=0) & (y<3)
print(k)
hist = fast_hist(x, y, 3)

print(hist)

def evaluate(hist):
    acc = np.diag(hist).sum() / hist.sum()
    acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist)+1e-10)
    mean_iu = np.nanmean(iu)
    freq = hist.sum(axis=1) / (hist.sum()+ 1e-10)
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    return acc, acc_cls, mean_iu, fwavacc

print(evaluate(hist))