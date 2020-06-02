import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from toolbox.utils import show_images

MNIST_TRANSFORM = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

class DigitSumImageLoader(Dataset):

    def __init__(self, data_dir, split, min_size, max_size, dataset_size, train = True, custom_transforms = None):
        self.data_dir = data_dir
        self.split = split
        self.min_size = min_size
        self.max_size = max_size
        self.dataset_size = dataset_size
        self.train = train
        self.transforms = self.get_transforms(custom_transforms)

        # Downloading MNIST Dataset into the folder of path data_dir.
        self.mnist = MNIST(data_dir, train = self.train, transform = self.transforms, download = True)
        mnist_len = self.mnist.__len__()
        mnist_items_range = np.arange(0, mnist_len)

        # Sampling the sets sizes randomly given min_size and max_size.
        set_sizes_range = np.arange(self.min_size, self.max_size + 1)
        set_sizes = np.random.choice(set_sizes_range, self.dataset_size, replace = True)
        self.mnist_items = []

        # Adding randomly sampled particles (MNIST images) into sets of the dataset.
        for i in range(self.dataset_size):
            set_items = np.random.choice(mnist_items_range, set_sizes[i], replace = True)
            self.mnist_items.append(set_items)

        self.data = [self.__getitem__(item) for item in range(self.__len__())]
        
    def __len__(self):
        return self.dataset_size

    def __getitem__(self, item):
        set_items = self.mnist_items[item]

        sum = 0
        img_list = []

        for mnist_item in set_items:
            img, label = self.mnist.__getitem__(mnist_item)
            sum += label
            img_list.append(img)
        
        out = torch.zeros(1,1)
        out[0][0] = sum
        
        return torch.stack(img_list, dim = 0), out

    def get_transforms(self, transforms):
        if transforms:
            return transforms
        else:
            return MNIST_TRANSFORM
    
    def read_images(self, item):
        set_items = self.__getitem__(item)[0]
        img_list = []

        for i in range(set_items.size()[0]):
            img = set_items[i]
            img_list.append(img.numpy()[0])
        
        show_images(img_list)
    
    def get_label(self, item):
        return self.__getitem__(item)[1]

'''
data_dir = '/Users/clementbeaulieu/Desktop/Polytechnique/UERecherche/DeepSets/data'
dataset = DigitSumImageDataset(data_dir, 'train', 1, 2, 5)

''''''
def mini_batch(batch_size, *tensors):
    if len(tensors) == 1:
        tensor = tensors[0]
        for i in range(0, len(tensor), batch_size):
            yield tensor[i:i+batch_size]
    else:
        for i in range(0, len(tensors[0]), batch_size):
            yield tuple(x[i:i+batch_size] for x in tensors)

data_loader = mini_batch(2, dataset)
''''''

def compute_batch(batch, model):#, args, model):
    target_list = []
    output_list = []
    for (input, target) in batch:
        input, target = input.to(args.device), target.to(args.device)
        target_list.append(target)
        output = model(input)
        output_list.append(output)
    output_batch = torch.stack(output_list, dim=0)
    target_batch = torch.stack(target_list, dim=0)
    return output_batch, target_batch

''''''
for i, batch in enumerate(data_loader):
    print(f'{i}, {batch}'
    
''''''
from models.digitsum_image import digitsum_image100

data_loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=lambda x: x)
model = digitsum_image100()

# iterate through the dataset:
for i, batch in enumerate(data_loader):
    print(f'{i}, {batch[0][0].size()}')
    #print(f'{i}, {batch[1][0].size()}')
    output_batch, target_batch = compute_batch(batch, model)
    print(output_batch, target_batch)

# iterate through the dataset:
for i, data in enumerate(dataset):
    print(f'{i}, {data[0].size()}')
    print(f'{i}, {data[1].size()}')'''