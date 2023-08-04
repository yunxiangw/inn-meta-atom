import numpy as np
from os.path import join
from scipy.io import loadmat
from torch.utils import data


def split_data():
    params = loadmat('./data/meta-atom/params.mat')['params'][:55000, :]
    targets = loadmat('./data/meta-atom/real_imag.mat')['output']

    x_train, y_train = params[:50000, :], targets[:50000, :]
    x_test, y_test = params[50000:, :], targets[50000:, :]

    np.save('./data/meta-atom/train/params.npy', x_train)
    np.save('./data/meta-atom/train/targets.npy', y_train)
    np.save('./data/meta-atom/test/params.npy', x_test)
    np.save('./data/meta-atom/test/targets.npy', y_test)


class MetasurfaceDataset(data.Dataset):
    def __init__(self, dataset_dir='./data/meta-atom', split='train'):
        # Set dataset directory and split.
        self.dataset_dir = dataset_dir
        self.split = split

        # Read the list of parameters and targets
        if self.split == 'train':
            self.path = join(self.dataset_dir, self.split)
        else:
            self.path = join(self.dataset_dir, self.split)

        self.params = np.load(join(self.path, 'params.npy')).astype(np.float32)
        self.targets = np.load(join(self.path, 'targets.npy')).astype(np.float32)

        # Rectangle: mean: 200, std: 92.3760, Meta-atom: mean: 175, std: 72.1688
        self.params = (self.params - 175) / 72.1688

        # self.targets = self.targets[:, :2]

    def __len__(self):
        return len(self.params)

    def __getitem__(self, index):
        return self.params[index], self.targets[index]


if __name__ == '__main__':
    split_data()
    # dataset = MetasurfaceDataset()
    # print(dataset[0])
