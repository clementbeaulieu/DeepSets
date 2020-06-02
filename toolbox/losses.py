import torch
import torch.nn as nn
import torch.nn.functional as F

def get_criterion(args):
    return{
        'mse': nn.MSELoss(reduction='mean')
    }[args.criterion]