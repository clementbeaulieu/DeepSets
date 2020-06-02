import numpy as np
import torch
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from torch import optim
from torch.autograd import Variable
from tqdm import tqdm

from loaders import DigitSumImageLoader
from models.digitsum_image import digitsum_image100


class SumOfDigits(object):
    def __init__(self, lr=1e-3, wd=5e-3):
        self.lr = lr
        self.wd = wd
        data_dir = '/Users/clementbeaulieu/Desktop/Polytechnique/UERecherche/DeepSets/data'
        log_dir = '/Users/clementbeaulieu/Desktop/Polytechnique/UERecherche/DeepSets/data/log'
        self.train_db = DigitSumImageLoader(data_dir, 'train', min_size=2, max_size=10, dataset_size=10000, train=True)
        self.test_db = DigitSumImageLoader(data_dir, 'test', min_size=5, max_size=50, dataset_size=10000, train=False)

        self.model = digitsum_image100()
        if torch.cuda.is_available():
            self.model.cuda()

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.wd)

        self.summary_writer = SummaryWriter(
            log_dir=log_dir)

    def train_1_epoch(self, epoch_num: int = 0):
        self.model.train()
        for i in tqdm(range(len(self.train_db))):
            loss = self.train_1_item(i)
            self.summary_writer.add_scalar('train_loss', loss, i + len(self.train_db) * epoch_num)

    def train_1_item(self, item_number: int) -> float:
        x, target = self.train_db.__getitem__(item_number)
        if torch.cuda.is_available():
            x, target = x.cuda(), target.cuda()

        x, target = Variable(x), Variable(target)

        self.optimizer.zero_grad()
        pred = self.model.forward(x)
        the_loss = F.mse_loss(pred, target)

        the_loss.backward()
        self.optimizer.step()

        the_loss_tensor = the_loss.data
        if torch.cuda.is_available():
            the_loss_tensor = the_loss_tensor.cpu()

        the_loss_numpy = the_loss_tensor.numpy().flatten()
        the_loss_float = float(the_loss_numpy[0])

        return the_loss_float

    def evaluate(self):
        self.model.eval()
        totals = [0] * 51
        corrects = [0] * 51

        for i in tqdm(range(len(self.test_db))):
            x, target = self.test_db.__getitem__(i)

            item_size = x.shape[0]

            if torch.cuda.is_available():
                x = x.cuda()

            pred = self.model.forward(Variable(x)).data

            if torch.cuda.is_available():
                pred = pred.cpu().numpy().flatten()

            pred = int(round(float(pred[0])))
            target = int(round(float(target.numpy()[0])))

            totals[item_size] += 1

            if pred == target:
                corrects[item_size] += 1

        totals = np.array(totals)
        corrects = np.array(corrects)

        print(corrects / totals)

def main():

    the_experiment = SumOfDigits(lr=1e-3)

    for i in range(20):
        the_experiment.train_1_epoch(i)
        the_experiment.evaluate()

if __name__ == '__main__':
    main()


'''
data_dir = '/Users/clementbeaulieu/Desktop/Polytechnique/UERecherche/DeepSets/data'
dataset = DigitSumImageDataset(data_dir, 'train', 1, 2, 5)

def mini_batch(batch_size, *tensors):
    if len(tensors) == 1:
        tensor = tensors[0]
        for i in range(0, len(tensor), batch_size):
            yield tensor[i:i+batch_size]
    else:
        for i in range(0, len(tensors[0]), batch_size):
            yield tuple(x[i:i+batch_size] for x in tensors)

data_loader = mini_batch(2, dataset)


def compute_batch(batch, model):#, args, model):
    target_list = []
    output_list = []
    batch_size=len(batch)
    target_size=batch[0][1].size()[0]
    for (input, target) in batch:
        #input, target = input.to(args.device), target.to(args.device)
        target_list.append(target)
        output = model(input)
        output_list.append(output)
    output_batch = torch.stack(output_list, dim=0)
    target_batch = torch.stack(target_list, dim=0)
    output_batch = output_batch.squeeze(1)
    target_batch = target_batch.squeeze(1)
    return output_batch, target_batch


for i, batch in enumerate(data_loader):
    print(f'{i}, {batch}'
    

from models.digitsum_image import digitsum_image100
from torch.utils.data import DataLoader

data_loader = DataLoader(dataset, batch_size=3, shuffle=False, collate_fn=lambda x: x)
model = digitsum_image100()

# iterate through the dataset:
for i, batch in enumerate(data_loader):
    print('batch idx: ', i)
    batch_size = len(batch)
    print('batch_size: ', batch_size)
    for j in range(batch_size):
        print('item ', j)
        print('input size :', batch[j][0].size())
        print('output :', batch[j][1])
    output_batch, target_batch = compute_batch(batch, model)
    print('output_batch, target_batch :')
    print(output_batch.size(), target_batch.size())

# iterate through the dataset:
for i, data in enumerate(dataset):
    print(f'{i}, {data[0].size()}')
    print(f'{i}, {data[1].size()}')
'''