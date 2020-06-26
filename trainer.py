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
        output_batch, target_batch = output_batch.to(args.device).requires_grad_(), target_batch.to(args.device)

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

    if args.val_type == 'classification':
        return validate_classification(args, val_loader, model, criterion, logger, epoch, eval_score, print_freq, tb_writer)
    
    #if args.val_type == 'regression':
    #    validate_regression(args, val_loader, model, criterion, logger, epoch, eval_score, print_freq, tb_writer)

    if args.val_type == 'digitsum':
        return validate_digitsum(args, val_loader, model, criterion, logger, epoch, eval_score, print_freq, tb_writer)


''' Validate function for classification validation-type.'''
def validate_classification(args, val_loader, model, criterion, logger, epoch, eval_score, print_freq, tb_writer):
    
    # switch to evaluate mode
    model.eval()
    meters = logger.reset_meters('val')
    end = time.time()

    hist = np.zeros((args.num_classes, args.num_classes))

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            batch_size = len(batch)

            meters['data_time'].update(time.time()-end, n=batch_size)

            output_batch, target_batch = utils.compute_batch(batch, args, model)
            output_batch, target_batch = output_batch.to(args.device).requires_grad_(), target_batch.to(args.device)

            label = target_batch.numpy()

            loss = criterion(output_batch, target_batch)
            meters['loss'].update(loss.data.item(), n=batch_size)

            # measure accuracy and record loss
            if eval_score is not None:
                acc1, pred, buff_label = eval_score(output_batch, target_batch)
                meters['acc1'].update(acc1, n=batch_size)
                meters['confusion_matrix'].update(pred.squeeze(), buff_label.type(torch.LongTensor))

                _, pred = torch.max(output_batch, 1)

                pred = pred.to('cpu').data.numpy()
                hist += metrics.fast_hist(pred.flatten(), label.flatten(), args.num_classes)
                mean_ap = round(np.nanmean(metrics.per_class_iu(hist)) * 100, 2)
                meters['mAP'].update(mean_ap, n=batch_size)

            # measure elapsed time
            meters['batch_time'].update(time.time() - end, n=batch_size)
            end = time.time()

            if i % print_freq == 0:
                print('Validation: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {score.val:.3f} ({score.avg:.3f})'.format(
                      i, len(val_loader), batch_time=meters['batch_time'], loss=meters['loss'],
                      score=meters['acc1']), flush=True)
        
    acc, acc_cls, mean_iu, fwavacc = metrics.evaluate(hist)
    meters['acc_class'].update(acc_cls)
    meters['meanIoU'].update(mean_iu)
    meters['fwavacc'].update(fwavacc)

    print(' * Validation set: Average loss {:.4f}, Accuracy {:.3f}%, Accuracy per class {:.3f}%, meanIoU {:.3f}%, \
            fwavacc {:.3f}% \n'.format(meters['loss'].avg, meters['acc1'].avg, meters['acc_class'].val,
                                       meters['meanIoU'].val, meters['fwavacc'].val ))

    logger.log_meters('val', n=epoch)
        
    if args.tensorboard:
        tb_writer.add_scalar('acc1/val', meters['acc1'].avg, epoch)
        tb_writer.add_scalar('loss/val', meters['loss'].avg, epoch)
        tb_writer.add_scalar('mAP/val', meters['mAP'].avg, epoch)
        tb_writer.add_scalar('acc_class/val', meters['acc_class'].val, epoch)
        tb_writer.add_scalar('meanIoU/val', meters['meanIoU'].val, epoch)
        tb_writer.add_scalar('fwavacc/val', meters['fwavacc'].val, epoch)
        
    return meters['mAP'].val, meters['loss'].avg


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

            meters['data_time'].update(time.time()-end, n=batch_size)

            output_batch, target_batch = utils.compute_batch(batch, args, model)
            output_batch, target_batch = output_batch.to(args.device).requires_grad_(), target_batch.to(args.device)

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

    
    set_mAP = metrics.set_mAP(meters, args.min_size_val, args.max_size_val)
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

'''Validation function for regression validation-type.'''
'''
def validate_regression(args, val_loader, model, criterion, logger, epoch, eval_score, print_freq, tb_writer):

    # switch to evaluate mode
    model.eval()
    meters = logger.reset_meters('val')
    end = time.time()

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            batch_size = len(batch)

            meters['data_time'].update(time.time()-end, n=batch_size)

            output_batch, target_batch = utils.compute_batch(batch, args, model)
            output_batch, target_batch = output_batch.to(args.device).requires_grad_(), target_batch.to(args.device)

            label_batch = target_batch.numpy()

            loss = criterion(output_batch, target_batch)
            meters['loss'].update(loss.data.item(), n=batch_size)
'''

#************************************************************#
#*************************** TEST ***************************#
#************************************************************#

def test(args, test_loader, model, criterion, epoch, eval_score=None, output_dir='pred', has_gt=True, print_freq=10):

    if args.test_type == 'classification':
        test_classification(args, test_loader, model, criterion, epoch, eval_score, output_dir, has_gt, print_freq)

    if args.test_type == 'digitsum':
        test_digitsum(args, test_loader, model, criterion, epoch, eval_score, output_dir, has_gt, print_freq)

''' Test function for classification test-type.'''
def test_classification(args, test_loader, model, criterion, epoch, eval_score, output_dir, has_gt, print_freq):

    model.eval()
    meters = metrics.make_meters(args.num_classes)
    end = time.time()
    hist = np.zeros((args.num_classes, args.num_classes))

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            batch_size = len(batch)

            meters['data_time'].update(time.time()-end, n=batch_size)

            output_batch, target_batch = utils.compute_batch(batch, args, model)
            output_batch, target_batch = output_batch.to(args.device).requires_grad_(), target_batch.to(args.device)

            label_batch = target_batch.numpy()

            loss = criterion(output_batch, target_batch)
            meters['loss'].update(loss.data.item(), n=batch_size)

            # measure accuracy and record loss
            if eval_score is not None:
                acc1, pred, buff_label = eval_score(output_batch, target_batch)
                meters['acc1'].update(acc1, n=batch_size)
                meters['confusion_matrix'].update(pred.squeeze(), buff_label.type(torch.LongTensor))

                _, pred = torch.max(output_batch, 1)

                pred = pred.to('cpu').data.numpy()
                hist += metrics.fast_hist(pred.flatten(), label_batch.flatten(), args.num_classes)
                mean_ap = round(np.nanmean(metrics.per_class_iu(hist)) * 100, 2)
                meters['mAP'].update(mean_ap, n=batch_size)

            # measure elapsed time
            meters['batch_time'].update(time.time() - end, n=batch_size)
            end = time.time()

            if i % print_freq == 0:
                print('Testing: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {score.val:.3f} ({score.avg:.3f})'.format(
                      i, len(test_loader), batch_time=meters['batch_time'], loss=meters['loss'],
                      score=meters['acc1']), flush=True)
    
    if eval_score is not None:
        acc, acc_cls, mean_iu, fwavacc = metrics.evaluate(hist)
        meters['acc_class'].update(acc_cls)
        meters['meanIoU'].update(mean_iu)
        meters['fwavacc'].update(fwavacc)

        print(' * Test set: Average loss {:.4f}, Accuracy {:.3f}%, Accuracy per class {:.3f}%, meanIoU {:.3f}%, \
            fwavacc {:.3f}% \n'.format(meters['loss'].avg, meters['acc1'].avg, meters['acc_class'].val,
                                       meters['meanIoU'].val, meters['fwavacc'].val ))
    
    metrics.save_meters(meters, os.path.join(args.log_dir, 'test_results_ep{}.json'.format(epoch)), epoch)



'''Test function for digitsum test-type (customized for digitsum experiment).'''
def test_digitsum(args, test_loader, model, criterion, epoch, eval_score, output_dir, has_gt, print_freq):

    # switch to evaluate mode
    model.eval()
    meters = metrics.make_meters(args.num_classes)
    end = time.time()

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            batch_size = len(batch)

            meters['data_time'].update(time.time()-end, n=batch_size)

            output_batch, target_batch = utils.compute_batch(batch, args, model)
            output_batch, target_batch = output_batch.to(args.device).requires_grad_(), target_batch.to(args.device)

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
                print('Testing: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {score.val:.3f} ({score.avg:.4f})'.format(
                      i, len(test_loader), batch_time=meters['batch_time'], loss=meters['loss'],
                      score=meters['acc1']), flush=True)
    
    if eval_score is not None:
        set_mAP = metrics.set_mAP(meters, args.min_size_val, args.max_size_val)
        if isinstance(set_mAP, torch.Tensor):
            set_mAP = set_mAP.cpu().data.item()
        meters['set_mAP'].update(set_mAP)
    
        print(' * Testing set: \t'
            'Average loss {:.4f}, Accuracy {:.3f}%\n'.format(meters['loss'].avg, meters['acc1'].avg))

        print('Accuracy per class of set size:')
        for i in range(args.min_size_val, args.max_size_val + 1):
            if (meters['set_class_total'].val[i] != 0):
                print('  Acc@ set size = {0} : {score:.2f} %'.format(i, score = 100 * meters['set_class_correct'].val[i] / meters['set_class_total'].val[i]))

        # convert numpy ndarrays to lists to be processed into json format in run.py
        class_correct = meters['set_class_correct'].val.tolist()
        class_total = meters['set_class_total'].val.tolist()
        meters['set_class_correct'].update(class_correct)
        meters['set_class_total'].update(class_total)

    metrics.save_meters(meters, os.path.join(args.log_dir, 'test_results_ep{}.json'.format(epoch)), epoch)