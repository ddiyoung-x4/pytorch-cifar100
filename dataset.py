import os
import numpy as np
import torch
from torch.utils.data import Dataset

def unpickle(file):

    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')

    return dict

class Cifar100Dataset(Dataset):
    def __init__(self, img_dir, train=True, transform=None, target_transform=None):
        self.img_dir = img_dir
        self.train = train
        if self.train:
            self.data_dict = unpickle(os.path.join(img_dir, 'train'))
        else:
            self.data_dict = unpickle(os.path.join(img_dir, 'test'))

        self.transform = transform
        self.target_transform = target_transform


    def __len__(self):
        return len(self.data_dict['fine_labels'.encode('utf-8')])

    def __getitem__(self, idx):
        image = self.data_dict['data'.encode('utf-8')][idx]
        r = image[:1024].reshape(32, 32)
        g = image[1024:2048].reshape(32, 32)
        b = image[2048:].reshape(32, 32)
        image = np.dstack((r, g, b))
        label = self.data_dict['fine_labels'.encode('utf-8')][idx]

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        return image, label

if __name__ == '__main__':
    from torchvision.transforms import ToTensor
    from torch.utils.data import DataLoader

    path = '/home/hun/shared/hdd_ext/nvme1/classification/cifar-100-python'
    train_dataset = Cifar100Dataset(path, train=True, transform=ToTensor())
    test_dataset = Cifar100Dataset(path, train=False, transform=ToTensor())

    train_loader = DataLoader(train_dataset, batch_size=64)
    test_loader = DataLoader(test_dataset, batch_size=64)

    for X, y in test_loader:
        print(f"Shape of X [N, C, H, W]: {X.shape}")
        print(f"Shape of y: {y.shape} {y.dtype}")
        break