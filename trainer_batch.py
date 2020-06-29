import time
import os
import numpy as np
import torch

from toolbox import utils, metrics


#************************************************************#
#************************* TRAINING *************************#
#************************************************************#

def train(args, train_loader, model, criterion, optimizer, logger, epoch, eval_score=None, print_freq=10, tb_writer=None):

    # Switch to train mode
    model.train()
    meters = logger.reset_meters('train')
    meters_params = logger.reset_meters('hyperparams')
    meters_params['learning_rate'].update(optimizer.param_groups[0]['lr'])
    end = time.time()

    for i, batch in enumerate(train_loader):

        batch_size = len(batch)

        # Measure data loading time.
        meters['data_time'].update(time.time()-end, n=batch_size)

        input_batch, target_batch = utils.batch(batch, args, model)

        output_batch = model(input_batch)

        output_batch, target_batch = output_batch.to(args.device), target_batch.to(args.device)

        loss = criterion(output_batch, target_batch)

        meters['loss'].update(loss.data.item(), n=batch_size)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if eval_score is not None:

            if args.train_type == 'classification':
                acc1, pred, label = eval_score(output_batch, target_batch)
                meters['acc1'].update(acc1, n=batch_size)
                meters['confusion_matrix'].update(pred.squeeze(), label.type(torch.LongTensor))

            if args.train_type == 'regression':
                mae, mse, rmse = eval_score(output_batch, target_batch)
                meters['mae'].update(mae.mean().item(), n=batch_size)
                meters['mse'].update(mse.mean().item(), n=batch_size)
                #meters['rmse'].update(rmse.mean().item(), n=batch_size)

        # measure elapsed time
        meters['batch_time'].update(time.time() - end, n=batch_size)
        end = time.time()

        if i % print_freq == 0:

            if args.train_type == 'classification':
                print('Epoch [{0}][{1}/{2}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Data {data_time.val:.3f} ({data_time.val:.3f})\t'
                    'LR {lr.val:.2e}\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(epoch, i, len(train_loader), batch_time=meters['batch_time'], data_time=meters['data_time'], lr=meters_params['learning_rate'], loss=meters['loss'], top1=meters['acc1']))
            
            if args.train_type == 'regression':
                print('Epoch [{0}][{1}/{2}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Data {data_time.val:.3f} ({data_time.val:.3f})\t'
                    'LR {lr.val:.2e}\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'MAE {mae.val:.3f} ({mae.avg:.3f})\t'
                    'MSE {mse.val:.3f} ({mse.avg:.3f})'.format(epoch, i, len(train_loader), batch_time=meters['batch_time'], data_time=meters['data_time'], lr=meters_params['learning_rate'], loss=meters['loss'], mae=meters['mae'], mse=meters['mse']))
    
    if args.tensorboard:
        if args.train_type == 'classification':
            tb_writer.add_scalar('acc1/train', meters['acc1'].avg, epoch)
            tb_writer.add_scalar('loss/train', meters['loss'].avg, epoch)
            tb_writer.add_scalar('learning rate', meters_params['learning_rate'].val, epoch)

        if args.train_type == 'regression':
            tb_writer.add_scalar('mae/train', meters['mae'].avg, epoch)
            tb_writer.add_scalar('mse/train', meters['mse'].avg, epoch)
            tb_writer.add_scalar('loss/train', meters['loss'].avg, epoch)
            tb_writer.add_scalar('learning rate', meters_params['learning_rate'].val, epoch)
       
    logger.log_meters('train', n=epoch)
    logger.log_meters('hyperparams', n=epoch)


#************************************************************#
#************************ VALIDATION ************************#
#************************************************************#


def validate(args, val_loader, model, criterion, logger, epoch, eval_score=None, print_freq=10, tb_writer=None):
    if args.val_type == 'digitsum':
        return validate_digitsum(args, val_loader, model, criterion, logger, epoch, eval_score, print_freq, tb_writer)


'''Validate function for digitsum validation-type (customized for digitsum experiment).'''
def validate_digitsum(args, val_loader, model, criterion, logger, epoch, eval_score, print_freq, tb_writer):

    # switch to evaluate mode
    model.eval()
    meters = logger.reset_meters('val')
    end = time.time()

    #correct = 0
    #total = 0
    #class_correct = torch.zeros(args.max_size_val + 1)
    #class_total = torch.zeros(args.max_size_val + 1)

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            batch_size = len(batch)

            # Measure data loading time.
            meters['data_time'].update(time.time()-end, n=batch_size)

            input_batch, target_batch = utils.batch(batch, args, model)

            output_batch = model(input_batch)
            output_batch, target_batch = output_batch.to(args.device), target_batch.to(args.device)

            loss = criterion(output_batch, target_batch)

            meters['loss'].update(loss.data.item(), n=batch_size)    

            # measure accuracy and record loss
            if eval_score is not None:
                input_sizes = utils.compute_input_sizes(batch)

                acc_batch, pred, buff_label = eval_score(output_batch, target_batch)

                # Update accuracy Acc1 on batch
                meters['acc1'].update(acc_batch, n=batch_size)

                # accuracy per class of set size

                class_correct_batch, class_total_batch = metrics.set_acc_class(pred, buff_label, batch_size, input_sizes, args.max_size_val)

                class_correct_batch = class_correct_batch.to('cpu').data.numpy()
                class_total_batch = class_total_batch.to('cpu').data.numpy()

                class_correct = meters['set_class_correct'].val
                class_total = meters['set_class_total'].val
                
                class_correct += class_correct_batch
                class_total += class_total_batch

                meters['set_class_correct'].update(class_correct)
                meters['set_class_total'].update(class_total)

            # measure elapsed time
            meters['batch_time'].update(time.time() - end, n=batch_size)
            end = time.time()

            if i % print_freq == 0:
                print('Validation: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {score.val:.3f} ({score.avg:.4f})'.format(
                      i, len(val_loader), batch_time=meters['batch_time'], loss=meters['loss'],
                      score=meters['acc1']), flush=True)

    #acc = correct/total

    
    set_mAP = metrics.set_mAP(meters, args.min_size_val, args.max_size_val, weight=args.set_weight)
    if isinstance(set_mAP, torch.Tensor):
        set_mAP = set_mAP.cpu().data.item()
    meters['set_mAP'].update(set_mAP)
    

    #meters['set_mAP'].update(meters['acc1'].avg)

    print(' * Validation set: \t'
        'Average loss {:.4f}, Accuracy {:.3f}\n'.format(meters['loss'].avg, meters['acc1'].avg))

    print('Accuracy per class of set size:')
    for i in range(args.min_size_val, args.max_size_val + 1):
        if (meters['set_class_total'].val[i] != 0):
            print('  Acc@ set size = {0} : {score:.2f} %'.format(i, score = 100 * meters['set_class_correct'].val[i] / meters['set_class_total'].val[i]))

    # convert numpy ndarrays to lists to be processed into json format in run.py
    class_correct = meters['set_class_correct'].val.tolist()
    class_total = meters['set_class_total'].val.tolist()
    meters['set_class_correct'].update(class_correct)
    meters['set_class_total'].update(class_total)

    logger.log_meters('val', n=epoch)

    if args.tensorboard:
        tb_writer.add_scalar('acc1/val', meters['acc1'].avg, epoch)
        tb_writer.add_scalar('loss/val', meters['loss'].avg, epoch)
        tb_writer.add_scalar('set_mAP/val', meters['set_mAP'].val, epoch)
        for i in range(args.min_size_val, args.max_size_val + 1):
            if (meters['set_class_total'].val[i] != 0):
                tb_writer.add_scalar('set_acc_class{}/val'.format(i), meters['set_class_correct'].val[i]/meters['set_class_total'].val[i], epoch)

    return meters['set_mAP'].val, meters['loss'].avg