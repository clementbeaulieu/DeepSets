import torch
from torch.utils.data import Dataset, DataLoader

'''
# dataset is a list of sequences/sentences
# the elements of the sentences could be anything, as long as it can be contained in a torch tensor
# usually, these will be indices of words based on some vocabulary
# 0 is commonly reserved for the padding token, here it appears once explicitly and on purpose,
#  to check that it functions properly (= in the same way as the automatically added padding tokens)
DATA = [
    [1, 2, 3],
    [4, 5],
    [6, 7, 8, 9],
    [4, 6, 2, 9, 0]
]
# need torch tensors for torch's pad_sequence(); this could be a part of e.g. dataset's __getitem__ instead
DATA = list(map(lambda x: torch.tensor(x), DATA))

print(DATA)
# vocab size (for embedding); including 0 (the padding token)
NUM_WORDS = 10

SEED = 0
# for consistent results between runs
torch.manual_seed(SEED)

BATCH_SIZE = 3
EMB_DIM = 2
LSTM_DIM = 5


class MinimalDataset(Dataset):
    def __init__(self, data) -> None:
        super().__init__()
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


dataset = MinimalDataset(DATA)
# len(data) is not divisible by batch_size on purpose to verify consistency across batch sizes
data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=lambda x: x)
# collate_fn is crucial for handling data points of varying length (as is the case here)
print(next(iter(data_loader)))
# I would think that we should always obtain:
# [ [1, 2, 3], [4, 5], [6, 7, 8, 9] ]
# but, without collate_fn set to identity as above, you would get:
# RuntimeError: invalid argument 0: Sizes of tensors must match except in dimension 0. Got 3 and 2 in dimension 1 ...
# ¯\_(ツ)_/¯

# iterate through the dataset:
for i, batch in enumerate(data_loader):
    print(f'{i}, {batch}')
'''
class DeepSetLoader(DataLoader):

    default_collate_fn = lambda x: x

    def __init__(self, dataset, batch_size=1, num_workers=0, shuffle=False, pin_memory=True, collate_fn=default_collate_fn):
        super().__init__(self, dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle, collate_fn=collate_fn)



def mini_batch(batch_size, *tensors):
    if len(tensors) == 1:
        tensor = tensors[0]
        for i in range(0, len(tensor), batch_size):
            yield tensor[i:i+batch_size]
    else:
        for i in range(0, len(tensors[0]), batch_size):
            yield tuple(x[i:i+batch_size] for x in tensors)
'''
mini = mini_batch(3, DATA)

for x in enumerate(mini):
    print(x)
'''