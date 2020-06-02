import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-7

def get_optim_parameters(model):
    for param in model.parameters():
        yield param

def get_optimizer(args, model):
    optimizer, scheduler = None, None

    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(get_optim_parameters(model), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(get_optim_parameters(model), lr=args.lr, amsgrad=False)
    else:
        raise 'Optimizer {} not available'.format(args.optimizer)

    if args.scheduler == 'StepLR':
        print(f' --- Setting lr scheduler to StepLR ---')
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step, gamma=args.lr_decay)
    elif args.scheduler == 'ExponentialLR':
        print(f' --- Setting lr scheduler to ExponentialLR ---')
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_decay)    
    elif args.scheduler == 'ReduceLROnPlateau':
        print(f' --- Setting lr scheduler to ReduceLROnPlateau ---') 
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=args.lr_decay, patience=args.step)
    else:
        raise f'Scheduler {args.scheduler} not available'
    
    return optimizer, scheduler

def adjust_learning_rate(args, optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every args.step epochs"""
    lr = args.lr * (0.1 ** (epoch // args.step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr