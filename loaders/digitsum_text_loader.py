import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class DigitSumTextLoader(Dataset):

    def __init__(self, data_dir, split, min_size, max_size, dataset_size, train = True, custom_transforms = None):
        self.data_dir = data_dir
        self.split = split
        self.min_size = min_size
        self.max_size = max_size
        self.dataset_size = dataset_size
        self.train = train

        digits = np.arange(0, 10)

        # Sampling the sets sizes randomly given min_size and max_size.
        set_sizes_range = np.arange(self.min_size, self.max_size + 1)
        set_sizes = np.random.choice(set_sizes_range, self.dataset_size, replace = True)
        self.digit_items = []

        # Adding randomly sampled particles (text digits) into sets of the dataset.
        for i in range(self.dataset_size):
            set_items = torch.from_numpy(np.random.choice(digits, set_sizes[i], replace = True)).type(torch.LongTensor).unsqueeze(1)
            self.digit_items.append(set_items)

        self.data = [self.__getitem__(item) for item in range(self.__len__())]

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, item):
        set_items = self.digit_items[item]

        out = np.array([[0.0]])
        digit_list = []

        for digit_item in set_items:
            out[0][0] += digit_item
            digit_list.append(digit_item)

        out = torch.from_numpy(out).type(torch.LongTensor)

        return torch.stack(digit_list, dim = 0).type(torch.LongTensor), out
    
    def get_label(self, item):
        return self.__getitem__(item)[1]