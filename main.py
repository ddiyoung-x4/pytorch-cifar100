import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

import argparse
import time
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from dataset import Cifar100Dataset
from utils import get_network, get_optimizer

import wandb

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int32(W * cut_rat)
    cut_h = np.int32(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def train(epoch):
    model.train()

    train_loss = 0.0

    for batch_idx, (images, labels) in enumerate(train_loader):
        if args.gpu:
            images = images.cuda()
            labels = labels.cuda()

        r = np.random.rand(1)
        if args.beta > 0 and r < args.cutmix_prob:
            lam = np.random.beta(args.beta, args.beta)
            rand_index = torch.randperm(images.size()[0]).cuda()
            target_a = labels
            target_b = labels[rand_index]
            bbx1, bby1, bbx2, bby2 = rand_bbox(images.size(), lam)
            images[:, :, bbx1:bbx2, bby1:bby2] = images[rand_index, :, bbx1:bbx2, bby1:bby2]
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (images.size()[-1] * images.size()[-2]))

            outputs = model(images)
            loss = criterion(outputs, target_a) * lam + criterion(outputs, target_b) * (1. - lam)
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
        train_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_loss = train_loss / len(train_loader.dataset)
    print(f'Training Epoch : {epoch} \tLoss: {train_loss:0.4f} ')
    wandb.log({"train_loss":train_loss})

@torch.no_grad()
def eval(epoch):

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
    test_acc = correct.float() / len(test_loader.dataset)
    print('Evaluating Network.....')
    print('Test set: Epoch: {}, Average loss: {:.4f}, Accuracy: {:.4f}, Time consumed:{:.2f}s'.format(
        epoch,
        test_loss / len(test_loader.dataset),
        test_acc,
        finish - start
    ))
    wandb.log({"test_loss":test_loss/len(test_loader.dataset), "test_acc": test_acc})


if __name__ == "__main__":

    def seed_everything(SEED):
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        torch.cuda.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.benchmark = True
    SEED = 42
    seed_everything(SEED)

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='resnet50', help='resnet model with layers 18, 35, 50, 101, 152')
    parser.add_argument('--batch-size', type=int, default=128, help='batch size for dataloader')
    parser.add_argument('--lr', type=float, default=0.1, help='initial learning rate')
    parser.add_argument('--optim', type=str, default='sgd', help='optimizer')
    parser.add_argument('--momentum', type=float, default=0.5, help='momentum')
    parser.add_argument('--w-decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train')
    parser.add_argument('--beta', type=float, default=0, help='hyperparameter beta')
    parser.add_argument('--cutmix-prob', type=float, default=0, help='cutmix probability')
    parser.add_argument('--gpu', default=False, help='use gpu or not')
    args = parser.parse_args()

    params = f"{args.model},batch_size-{args.batch_size},lr-{args.lr},optim-{args.optim},momentum-{args.momentum}"
    wandb.init(project="cifar100", entity='ddiyoung-x4', name=params, settings=wandb.Settings(_disable_stats=True))

    trans = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])

    test_trans = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])
    # server3
    path = '/home/hun/shared/hdd_ext/nvme1/classification/cifar-100-python'
    # server4
    # path = '/home/hun/shared/hdd_ext/nvme1/Cifar100/cifar-100-python'
    train_dataset = Cifar100Dataset(path, train=True, transform=trans)
    test_dataset = Cifar100Dataset(path, train=False, transform=test_trans)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    if args.gpu:
        model = get_network(args).cuda()
    else:
        model = get_network(args)

    criterion = nn.CrossEntropyLoss()
    optimizer = get_optimizer(args, model)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.1, pct_start=0.1, total_steps=args.epochs, steps_per_epoch=len(train_loader), epochs=args.epochs)

    for epoch in range(args.epochs):
        train(epoch)
        eval(epoch)
        scheduler.step()
