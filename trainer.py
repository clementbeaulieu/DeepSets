import time
import os
import numpy as np
import torch

from toolbox import utils, metrics


def train(args, train_loader, model, criterion, optimizer, logger, epoch, eval_score=None, print_freq=10, tb_writer=None):

    # Switch to train mode
    model.train()
    meters = logger.reset_meters('train')
    meters_params = logger.reset_meters('hyperparams')
    meters_params['learning_rate'].update(optimizer.param_groups[0]['lr'])
    end = time.time()

    #print("train_loader first item")
    #print(train_loader[0])

    for i, batch in enumerate(train_loader):
        #print(f'{i} - {input.size()} - {target.size()}')

        batch_size = len(batch)

        #print('i: ', i)
        #print('batch_size: ', batch_size)
        #print('batch: ', batch)

        # Measure data loading time.
        meters['data_time'].update(time.time()-end, n=batch_size)

        output_batch, target_batch = utils.compute_batch(batch, args, model)

        output_batch, target_batch = output_batch.to(args.device), target_batch.to(args.device)

        #print('output_batch: ', output_batch)
        #print('ouput_batch size: ', output_batch.size())
        #print('target_batch: ', target_batch)
        #print('target_batch size: ', target_batch.size())

        loss = criterion(output_batch, target_batch)

        meters['loss'].update(loss.data.item(), n=batch_size)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if eval_score is not None:

            if args.task_type == 'classification':
                acc1, pred, label = eval_score(output_batch, target_batch)
                meters['acc1'].update(acc1, n=batch_size)
                #meters['confusion_matrix'].update(
                #    pred.squeeze(), label.type(torch.LongTensor))

            if args.task_type == 'regression':
                mae, mse, rmse = eval_score(output_batch, target_batch)
                R = 1 - mse.mean().item()/mse.std().pow(2).item()
                meters['acc1'].update(R, n=batch_size)

            if args.task_type == 'custom':
                acc1 = eval_score(output_batch, target_batch)
                meters['acc1'].update(acc1, n=batch_size)

        # measure elapsed time
        meters['batch_time'].update(time.time() - end, n=batch_size)
        end = time.time()

        if i % print_freq == 0:
            print('Epoch [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.val:.3f})\t'
                  'LR {lr.val:.2e}\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(epoch, i, len(train_loader), batch_time=meters['batch_time'], data_time=meters['data_time'], lr=meters_params['learning_rate'], loss=meters['loss'], top1=meters['acc1']))

    if args.tensorboard:
        tb_writer.add_scalar('acc1/train', meters['acc1'].avg, epoch)
        tb_writer.add_scalar('loss/train', meters['loss'].avg, epoch)
        tb_writer.add_scalar('learning rate', meters_params['learning_rate'].val, epoch)
       
    logger.log_meters('train', n=epoch)
    logger.log_meters('hyperparams', n=epoch)

def validate(args, val_loader, model, criterion, logger, epoch, eval_score=None, print_freq=10,tb_writer=None):

    # switch model to evaluate
    model.eval()
    totals = [0] * 51
    corrects = [0] * 51

    for i, batch in enumerate(val_loader):
        x, target = batch[0]

        item_size = x.shape[0]

        if torch.cuda.is_available():
            x = x.cuda()

        pred = model.forward(x).data

        if torch.cuda.is_available():
            pred = pred.cpu().numpy().flatten()

        pred = int(round(float(pred[0])))
        target = int(round(float(target.numpy()[0])))

        totals[item_size] += 1

        if pred == target:
            corrects[item_size] += 1

    totals = np.array(totals)
    corrects = np.array(corrects)

    print(corrects)
