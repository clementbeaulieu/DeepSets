import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

import h5py

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

def load_h5(h5_filename):
    f = h5py.File(h5_filename,'r')
    data = f['data'][:]
    label = f['label'][:]
    return (data, label)

def load_data(data_dir):
    data_train0, label_train0 = load_h5(os.path.join(data_dir, 'ply_data_train0.h5'))
    data_train1, label_train1 = load_h5(os.path.join(data_dir, 'ply_data_train1.h5'))
    data_train2, label_train2 = load_h5(os.path.join(data_dir, 'ply_data_train2.h5'))
    data_train3, label_train3= load_h5(os.path.join(data_dir, 'ply_data_train3.h5'))
    data_train4, label_train4 = load_h5(os.path.join(data_dir, 'ply_data_train4.h5'))

    data_test0, label_test0 = load_h5(os.path.join(data_dir, 'ply_data_test0.h5'))
    data_test1, label_test1 = load_h5(os.path.join(data_dir, 'ply_data_test1.h5'))

    train_data = np.concatenate([data_train0,data_train1,data_train2,data_train3,data_train4])
    train_label = np.concatenate([label_train0,label_train1,label_train2,label_train3,label_train4])

    test_data = np.concatenate([data_test0,data_test1])
    test_label = np.concatenate([label_test0,label_test1])

    return train_data, train_label, test_data, test_label

def sample_cloud(data, nb_points, seed=None):
    if seed is not None:
        np.random.seed(seed)

    N = data.shape[0]
    M = data.shape[1]

    sampled_data = np.zeros((N, nb_points, 3))

    for i in range(N):
        idx = np.random.choice(M, nb_points, replace=False)
        sampled_data[i] = data[i][idx][:]
    
    return sampled_data

def split(data, dataset_size, seed=None):
    N = data.shape[0]

    if seed is not None:
        np.random.seed(seed)
    
    idx = np.random.choice(N, dataset_size, replace=False)
    sampled_data = data[idx][:][:]

    return sampled_data

def rotate_z(theta, x):
    theta = np.expand_dims(theta, 1)
    outz = np.expand_dims(x[:,:,2], 2)
    sin_t = np.sin(theta)
    cos_t = np.cos(theta)
    xx = np.expand_dims(x[:,:,0], 2)
    yy = np.expand_dims(x[:,:,1], 2)
    outx = cos_t * xx - sin_t*yy
    outy = sin_t * xx + cos_t*yy
    return np.concatenate([outx, outy, outz], axis=2)
    
def augment(x):
    bs = x.shape[0]
    #rotation
    thetas = np.random.uniform(-0.1, 0.1, [bs,1])*np.pi
    rotated = rotate_z(thetas, x)
    #scaling
    scale = np.random.rand(bs,1,3)*0.45 + 0.8
    return rotated*scale

def standardize(x):
    clipper = np.mean(np.abs(x), (1,2), keepdims=True)
    z = np.clip(x, -100*clipper, 100*clipper)
    mean = np.mean(z, (1,2), keepdims=True)
    std = np.std(z, (1,2), keepdims=True)
    return (z-mean)/std

class CloudPointsLoader(Dataset):

    def __init__(self, data_dir, split, num_classes=40, nb_points=1000, train=True, do_standardize=True, do_augmentation=False, seed=None):
        self.data_dir = data_dir
        self.num_classes = num_classes
        self.nb_points = nb_points
        self.train = train
        
        if train:
            self.data, self.label, _ , _ = load_data(data_dir)
            self.data = sample_cloud(self.data, self.nb_points, seed)
        else:
            _, _, self.data, self.label = load_data(data_dir)
            self.data = sample_cloud(self.data, self.nb_points, seed)









'''
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
        ''' '''reset order of h5 files''' '''
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
        ''''''returned dimension may be smaller than self.batch_size ''''''
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
'''