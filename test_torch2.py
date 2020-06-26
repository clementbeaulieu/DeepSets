import torch

x = torch.zeros((4, 10))
x[0][6] = 4.0
x[1][6] = 8.0
x[0][8] = 5.0
x[3][8] = 2.0

M = x.size(0)
one_tensor = torch.ones(M,1)
y = maxpool(x)
ym = torch.matmul(one_tensor, y)

def set_mAP(meters, min_size, max_size, weight='mean'):
    set_class_correct = meters['set_class_correct'].val
    set_class_total = meters['set_class_total'].val
    
    set_mAP = 0.0

    length = max_size - min_size + 1

    if weight == 'mean':
        mean_weight = torch.zeros(max_size+1)
        for i in range(min_size, max_size+1):
            mean_weight = 1.0/length
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
            set_mAP += weight[i].data.item() * set_class_correct[i]/set_class_total[i]

    return set_mAP