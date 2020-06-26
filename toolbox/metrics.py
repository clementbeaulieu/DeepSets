import torch
import numpy as np
import math
import json

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def value(self):
        return self.avg

class SumMeter(object):
    """Computes and stores the sum and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n

    def value(self):
        return self.sum


class ValueMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0

    def update(self, val):
        self.val = val

    def value(self):
        return self.val



class ConfusionMeter(object):
    """
    The ConfusionMeter constructs a confusion matrix for a multi-class
    classification problems. It does not support multi-label, multi-class problems:
    for such problems, please use MultiLabelConfusionMeter.
    """

    def __init__(self, k, normalized=False):
        """
        Args:
            k (int): number of classes in the classification problem
            normalized (boolean): Determines whether or not the confusion matrix
                is normalized or not
        """
        self.conf = np.ndarray((k, k), dtype=np.int32)
        self.normalized = normalized
        self.k = k
        self.reset()

    def reset(self):
        self.conf.fill(0)

    def update(self, predicted, target):
        """
        Computes the confusion matrix of K x K size where K is no of classes
        Args:
            predicted (tensor): Can be an N x K tensor of predicted scores obtained from
                the model for N examples and K classes or an N-tensor of
                integer values between 0 and K-1.
            target (tensor): Can be a N-tensor of integer values assumed to be integer
                values between 0 and K-1 or N x K tensor, where targets are
                assumed to be provided as one-hot vectors
        """
        if predicted.numel()>1:
            predicted.squeeze_()
            target.squeeze_()
        else:
            predicted = predicted.view(-1)
            target = target.view(-1)

        predicted = predicted.to('cpu').numpy()
        target = target.to('cpu').numpy()

        assert predicted.shape[0] == target.shape[0], \
            'number of targets and predicted outputs do not match'

        if np.ndim(predicted) != 1:
            assert predicted.shape[1] == self.k, \
                'number of predictions does not match size of confusion matrix'
            predicted = np.argmax(predicted, 1)
        else:
            assert (predicted.max() < self.k) and (predicted.min() >= 0), \
                'predicted values are not between 1 and k'

        onehot_target = np.ndim(target) != 1
        if onehot_target:
            assert target.shape[1] == self.k, \
                'Onehot target does not match size of confusion matrix'
            assert (target >= 0).all() and (target <= 1).all(), \
                'in one-hot encoding, target values should be 0 or 1'
            assert (target.sum(1) == 1).all(), \
                'multi-label setting is not supported'
            target = np.argmax(target, 1)
        else:
            assert (predicted.max() < self.k) and (predicted.min() >= 0), \
                'predicted values are not between 0 and k-1'

        # hack for bincounting 2 arrays together
        x = predicted + self.k * target
        bincount_2d = np.bincount(x.astype(np.int32),
                                  minlength=self.k ** 2)
        assert bincount_2d.size == self.k ** 2
        conf = bincount_2d.reshape((self.k, self.k))

        self.conf += conf

    def value(self):
        """
        Returns:
            Confustion matrix of K rows and K columns, where rows corresponds
            to ground-truth targets and columns corresponds to predicted
            targets.
        """
        if self.normalized:
            conf = self.conf.astype(np.float32)
            res = conf / conf.sum(1).clip(min=1e-12)[:, None]
            return res.tolist()
        else:
            return self.conf.tolist()

def make_meters(num_classes=2):
    meters_dict = {
        'loss': AverageMeter(),
        'acc1': AverageMeter(),
        'mae': AverageMeter(),
        'mse': AverageMeter(),
        'rmse': AverageMeter(),
        'set_class_correct': ValueMeter(), # number of correct instances per class of set size
        'set_class_total': ValueMeter(), # total of instances per class of set size
        'set_mAP': ValueMeter(), # mean Average Precision for the classes of set sizes
        'mAP': AverageMeter(),
        'meanIoU': ValueMeter(),
        'acc_class': ValueMeter(),
        'fwavacc': ValueMeter(),
        'batch_time': AverageMeter(),
        'data_time': AverageMeter(),
        'epoch_time': SumMeter(),
        'confusion_matrix': ConfusionMeter(num_classes),
    }
    return meters_dict

def save_meters(meters, fn, epoch=0):

    logged = {}
    for name, meter in meters.items():
        logged[name] = meter.value()

    if epoch > 0:
        logged['epoch'] = epoch

    print(f'Saving meters to {fn}')
    with open(fn, 'w') as f:
        json.dump(logged, f)

def accuracy_classif(output, target, topk=1):
    """Computes the precision@k for the specified values of k"""
    maxk = 1
    #maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    # res = []
    correct_k = correct[:maxk].view(-1).float().sum(0)
    correct_k.mul_(1.0 / batch_size)
    res = correct_k.clone()

    return res.item(), pred, target

def accuracy_regression(output, target):
    '''
    mae_mean = (output - target).abs().mean()
    mae_std = (output - target).abs().std()
    mse_mean = (output - target).pow(2).mean()
    mse_std = (output -target).pow(2).std()
    rmse_mean = (output - target).pow(2).mean().sqrt()
    rmse_std = (output - target).pow(2).std().sqrt()
    return mae, mse, rmse
    return mae_mean.item(), mae_std.item(), mse_mean.item(), mse_std.item(), rmse_mean.item(), rmse_std.item()
    '''
    mae = (output - target).abs()
    mse = (output - target).pow(2)
    rmse = (output - target).pow(2).sqrt()
    return mae, mse, rmse

def evaluate(hist):
    acc = np.diag(hist).sum() / hist.sum()
    acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist)+1e-10)
    mean_iu = np.nanmean(iu)
    freq = hist.sum(axis=1) / (hist.sum()+ 1e-10)
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    return acc, acc_cls, mean_iu, fwavacc


def fast_hist(pred, label, n):
    k = (label >= 0) & (label < n)
    return np.bincount(
        n * label[k].astype(int) + pred[k], minlength=n ** 2).reshape(n, n)

def per_class_iu(hist):
    return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist)+ 1e-10)

def digitsum_score(output, target): #set_score
    output = output.data
    target = target.data
    
    pred = torch.round(output)
    buff_label = torch.round(target)

    batch_size = target.size(0)
    correct_batch = (pred == buff_label).sum().double().item()

    acc_batch = correct_batch/batch_size

    return acc_batch, pred, buff_label

def set_acc_class(pred, buff_label, batch_size, input_sizes, max_size):
    class_correct_batch = torch.zeros(max_size + 1)
    class_total_batch = torch.zeros(max_size + 1)
    c = (pred == buff_label)
    for j in range(batch_size):
        item_size = input_sizes[j]
        class_correct_batch[item_size] += c[j].item()
        class_total_batch[item_size] += 1.0
    return class_correct_batch, class_total_batch

def set_mAP(meters, min_size, max_size, weight='mean'):
    set_class_correct = meters['set_class_correct'].val
    set_class_total = meters['set_class_total'].val
    
    set_mAP = 0.0

    length = max_size - min_size + 1

    if weight == 'mean':
        mean_weight = torch.zeros(max_size+1)
        for i in range(min_size, max_size+1):
            mean_weight[i] = 1.0/length
        weight = mean_weight

    if weight == 'linear':
        linear_weight = torch.zeros(max_size+1)
        for i in range(min_size, max_size+1):
            linear_weight[i] = - 1.0/(length * (length - 1)) * (i - min_size) + 3.0 / (2.0 * length)
        weight = linear_weight

    if weight == 'exp':
        exp_weight = torch.zeros(max_size+1)
        gamma = 1.0/max_size
        alpha = (math.exp(- gamma * (max_size+1)) - math.exp(- gamma * min_size))/(math.exp(- gamma) - 1.0)
        for i in range(min_size, max_size+1):
            exp_weight[i] = math.exp(- gamma * i)/alpha
        weight = exp_weight

    for i in range(min_size, max_size + 1):
        if (set_class_total[i] != 0):
            set_mAP += 1.0 * set_class_correct[i]/set_class_total[i]

    return set_mAP

'''Return the corresponding score to the task-type.'''
def get_score(eval_score):
    return{
        'regression': accuracy_regression,
        'classification': accuracy_classif,
        'digitsum': digitsum_score,
    }[eval_score]