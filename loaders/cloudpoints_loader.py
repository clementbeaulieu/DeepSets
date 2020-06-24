import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

idx2label = {
    0: 'airplane',
    1: 'bathtub',
    2: 'bed',
    3: 'bench',
    4: 'bookshelf',
    5: 'bottle',
    6: 'bowl',
    7: 'car',
    8: 'chair',
    9: 'cone',
    10: 'cup',
    11: 'curtain',
    12: 'desk',
    13: 'door',
    14: 'dresser',
    15: 'flower_pot',
    16: 'glass_box',
    17: 'guitar',
    18: 'keyboard',
    19: 'lamp',
    20: 'laptop',
    21: 'mantel',
    22: 'monitor',
    23: 'night_stand',
    24: 'person',
    25: 'piano',
    26: 'plant',
    27: 'radio',
    28: 'range_hood',
    29: 'sink',
    30: 'sofa',
    31: 'stairs',
    32: 'stool',
    33: 'table',
    34: 'tent',
    35: 'toilet',
    36: 'tv_stand',
    37: 'vase',
    38: 'wardrobe',
    39: 'xbox' 
}

classnames = [v for k,v in idx2label.items()]

class CloudPointsLoader(Dataset):

    def __init__(self, data_dir, split, num_classes=40, train = True, first_subsampling_dl=0.03, config=None, data_augmentation=True)):
        
        self.data_dir = data_dir
        self.split = split

        self.config = config
        self.first_subsampling_dl = first_subsampling_dl

        self.idx2label = idx2label
        self.classnames = classnames

        self.data_augmentation = data_augmentation
        self.points, self.normals, self.labels = [], [], []

        self.num_classes = num_classes
        self.train = train
        
        # Load wanted points if possible
        print(f'\nLoading {self.split} points')
        filename = os.path.join(self.data_dir, f'{self.split}_{first_subsampling_dl:.3f}_record.pkl')
        if os.path.exists(filename):
            with open(filename, 'rb') as file:
                self.points, self.normals, self.labels = pickle.load(file)
        else:



def shuffle_data(data, labels):
    """ Shuffle data and labels.
        Input:
          data: B,N,... numpy array
          label: B,... numpy array
        Return:
          shuffled data, label and shuffle indices
    """
    idx = np.arange(len(labels))
    np.random.shuffle(idx)
    return data[idx, ...], labels[idx], idx


def getDataFiles(list_filename):
    return [line.rstrip() for line in open(list_filename)]


def load_h5(h5_filename):
    f = h5py.File(h5_filename)
    data = f['data'][:]
    label = f['label'][:]
    return (data, label)


def loadDataFile(filename):
    return load_h5(filename)


class ModelNetH5Dataset(object):
    def __init__(self, data_dir, batch_size=32, npoints=1024, shuffle=True,train=False):
        self.data_dir=data_dir
        if(train):
            self.list_filename = data_dir+'train_files.txt'
        else:
            self.list_filename = data_dir+'test_files.txt'
        self.batch_size = batch_size
        self.npoints = npoints
        self.shuffle = shuffle
        self.h5_files = getDataFiles(self.list_filename)
        self.reset()
    def reset(self):
        ''' reset order of h5 files '''
        self.file_idxs = np.arange(0, len(self.h5_files))
        if self.shuffle:
            np.random.shuffle(self.file_idxs)
        self.current_data = None
        self.current_label = None
        self.current_file_idx = 0
        self.batch_idx = 0

#    def _augment_batch_data(self, batch_data):
#        rotated_data = provider.rotate_point_cloud(batch_data)
##        rotated_data = provider.rotate_perturbation_point_cloud(rotated_data)
##        jittered_data = provider.random_scale_point_cloud(rotated_data[:, :, 0:3])
##        jittered_data = provider.shift_point_cloud(jittered_data)
##        jittered_data = provider.jitter_point_cloud(jittered_data)
##        rotated_data[:, :, 0:3] = jittered_data
#        return provider.shuffle_points(rotated_data)

    def _get_data_filename(self):
        return self.h5_files[self.file_idxs[self.current_file_idx]]

    def _load_data_file(self, filename):
        self.current_data, self.current_label = load_h5(filename)
        self.current_label = np.squeeze(self.current_label)
        self.batch_idx = 0      
        
        if self.shuffle:
            self.current_data, self.current_label, _ = shuffle_data(
                self.current_data, self.current_label)

    def _has_next_batch_in_file(self):
        return self.batch_idx*self.batch_size < self.current_data.shape[0]

    def num_channel(self):
        return 3

    def has_next_batch(self):
        # TODO: add backend thread to load data
        if (self.current_data is None) or (not self._has_next_batch_in_file()):
            if self.current_file_idx >= len(self.h5_files):
                return False
            self._load_data_file(self.root+self._get_data_filename())
            self.batch_idx = 0
            self.current_file_idx += 1
        return self._has_next_batch_in_file()

    def next_batch(self, augment=False):
        ''' returned dimension may be smaller than self.batch_size '''
        start_idx = self.batch_idx * self.batch_size
        end_idx = min((self.batch_idx+1) * self.batch_size, self.current_data.shape[0])
#        bsize = end_idx - start_idx
#        batch_label = np.zeros((bsize), dtype=np.int32)
        data_batch = self.current_data[start_idx:end_idx, 0:self.npoints, :].copy()
        label_batch = self.current_label[start_idx:end_idx].copy()
        self.batch_idx += 1
#        if augment:
#            data_batch = self._augment_batch_data(data_batch)
        return data_batch, label_batch
    