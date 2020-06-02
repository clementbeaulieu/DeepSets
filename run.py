import os
import sys
import shutil

import torch

from loaders import get_loader
from models import get_model
from toolbox import utils, metrics, logger, losses, optimizers
import trainer as trainer
from args import parse_args

from torch.utils.tensorboard import SummaryWriter

'''
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
'''

def init_logger(args, model):
    # set loggers
    exp_name = args.name
    exp_logger = logger.Experiment(exp_name, args.__dict__)
    exp_logger.add_meters('train', metrics.make_meters(args.num_classes))
    exp_logger.add_meters('val', metrics.make_meters(args.num_classes))
    exp_logger.add_meters(
        'hyperparams', {'learning_rate': metrics.ValueMeter()})
    return exp_logger


def main():
    global args

    if len(sys.argv) > 1:
        args = parse_args()
        print('----- Experiments parameters -----')
        for k, v in args.__dict__.items():
            print(k, ':', v)
    else:
        print('Please provide some parameters for the current experiment. Check-out arg.py for more info!')
        sys.exit()

    # init tensorboard summary is asked
    tb_writer = SummaryWriter(f'{args.data_dir}/runs/{args.name}/tensorboard') if args.tensorboard else None

    # init data loaders
    loader = get_loader(args)
    train_loader = torch.utils.data.DataLoader(loader(data_dir=args.data_dir, split='train', min_size=args.min_size, max_size=args.max_size,
                                                      dataset_size=args.dataset_size), batch_size=args.batch_size, shuffle=True, num_workers=args.workers, collate_fn=lambda x: x, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(loader(data_dir=args.data_dir, split='val', min_size=5,
                                                    max_size=50, dataset_size=args.dataset_size), batch_size=1, shuffle=False, num_workers=args.workers, collate_fn=lambda x: x, pin_memory=True)

    exp_logger, lr = None, None

    model = get_model(args)
    criterion = losses.get_criterion(args)

    if exp_logger is None:
        exp_logger = init_logger(args, model)

    optimizer, scheduler = optimizers.get_optimizer(args, model)

    print('  + Number of params: {}'.format(utils.count_params(model)))

    model.to(args.device)
    criterion.to(args.device)

    '''
    if args.test:
        test_loader = torch.utils.data.DataLoader(loader(data_dir=args.data_dir, split='test',
                                                         phase='test', out_name=True, num_classes=args.num_classes), batch_size=args.batch_size,
                                                  shuffle=False, num_workers=args.workers, pin_memory=True)
        trainer.test(args, test_loader, model, criterion, args.start_epoch,
                     eval_score=metrics.accuracy_classif, output_dir=args.out_pred_dir, has_gt=True)
        sys.exit()
    '''

    #is_best = True

    for epoch in range(args.start_epoch, args.epochs + 1):
        print('Current epoch:', epoch)

        trainer.train(args, train_loader, model, criterion, optimizer, exp_logger, epoch, eval_score=metrics.accuracy_regression, tb_writer=tb_writer)

        # evaluate on validation set
        #trainer.validate(args, train_loader, model, criterion, optimizer, exp_logger, epoch, eval_score=None, tb_writer=tb_writer)

        # Update learning rate
        if scheduler is None:
            trainer.adjust_learning_rate(args, optimizer, epoch)
        else:
            prev_lr =  optimizer.param_groups[0]['lr']
            if 'ReduceLROnPlateau' == args.scheduler:
                scheduler.step(val_loss)
            else:    
                scheduler.step()
                
            print(f"Updating learning rate from {prev_lr} to {optimizer.param_groups[0]['lr']}")

    trainer.validate(args, val_loader, model, criterion, logger, epoch, eval_score=None, print_freq=10,tb_writer=None)
    
    if args.tensorboard:
        tb_writer.close()

    print("That's all folks!")

if __name__ == '__main__':
    main()