import argparse
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from dataset import Cifar100Dataset
from utils import get_network

def train(epoch):
    model.train()
    for batch_idx, (images, labels) in enumerate(train_loader):
        if args.gpu:
            images = images.cuda()
            labels = labels.cuda()

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        trained_samples = batch_idx * args.batch_size + len(images)
        total_samples = len(train_loader.dataset)
        print(f'Training Epoch : {epoch} [{trained_samples}/{total_samples}]\tLoss: {loss.item():0.4f} ')

def eval(epoch=0):

    start = time.time()
    model.eval()

    test_loss = 0.0
    correct = 0.0

    for images, labels in test_loader:
        if args.gpu:
            images = images.cuda()
            labels = labels.cuda()

        outputs = model(images)
        loss = criterion(outputs, labels)

        test_loss += loss.item()
        _, predicts = outputs.max(1)
        correct += predicts.eq(labels).sum()

    finish = time.time()

    print('Evaluating Network.....')
    print('Test set: Epoch: {}, Average loss: {:.4f}, Accuracy: {:.4f}, Time consumed:{:.2f}s'.format(
        epoch,
        test_loss / len(test_loader.dataset),
        correct.float() / len(test_loader.dataset),
        finish - start
    ))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='resnet50', help='resnet model with layers 18, 35, 50, 101, 152')
    parser.add_argument('--batch-size', type=int, default=64, help='batch size for dataloader')
    parser.add_argument('--lr', type=int, default=0.1, help='initial learning rate')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train')
    parser.add_argument('--gpu', default=False, help='use gpu or not')
    args = parser.parse_args()

    trans = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize()
    ])

    test_trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize()
    ])
    path = '/home/hun/shared/hdd_ext/nvme1/classification/cifar-100-python'
    train_dataset = Cifar100Dataset(path, train=True, transform=trans)
    test_dataset = Cifar100Dataset(path, train=False, transform=test_trans)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    model = get_network(args)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

    for epoch in range(args.epochs):
        train(epoch)
        eval(epoch)
