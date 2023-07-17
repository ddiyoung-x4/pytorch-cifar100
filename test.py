import argparse
import os

os.environ["CUDA_DEVICE_ORDER"] ="PCI_BUS_ID"

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

import dataset
from utils import get_network

def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(k=maxk, dim=1, largest=True, sorted=True)
    correct = pred.eq(target.view(target.size(0), -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:, :k].sum()
        res.append(correct_k)

    return res

def eval(epoch):

    start = time.time()
    model.eval()

    test_loss = 0.0
    correct_top1 = 0.0
    correct_top5 = 0.0

    for images, labels in test_loader:
        if args.gpu:
            images = images.cuda()
            labels = labels.cuda()

        outputs = model(images)
        loss = criterion(outputs, labels)

        test_loss += loss.item()

        # top-1, top-5 ACC
        top1, top5 = accuracy(outputs, labels, topk=(1, 5))
        correct_top1 += top1
        correct_top5 += top5

    finish = time.time()
    test_top1_acc = correct_top1.float() / len(test_loader.dataset)
    test_top5_acc = correct_top5.float() / len(test_loader.dataset)
    print('Evaluating Network.....')
    print('Test set: Epoch: {}, Average loss: {:.4f}, Top1 Accuracy: {:.4f}, Top5 Accuracy: {:.4f} Time consumed:{:.2f}s'.format(
        epoch,
        test_loss / len(test_loader.dataset),
        test_top1_acc,
        test_top5_acc,
        finish - start
    ))
    # wandb.log({"test_loss":test_loss/len(test_loader.dataset), "test_top1_acc": test_top1_acc, "test_top5_acc": test_top5_acc}, step=epoch)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='model network type')
    parser.add_argument('--weights', type=str, required=True, help='the weight file(pt, pth) of model')
    parser.add_argument('--batch-size', type=int, required=True, help='batch size for dataloader')
    parser.add_argument('--gpu', default=False, help='use gpu or not')
    args = parser.parse_args()

    path = 'your_cifar100_path'

    trans = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])

    test_dataset = dataset.Cifar100Dataset(path, train=False, transform=trans)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    if args.gpu:
        model = get_network(args).cuda()
    else:
        model = get_network(args)
    model = torch.load(args.weights)

    correct_top1 = 0.0
    correct_top5 = 0.0

    model.eval()
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(test_loader):
            if args.gpu:
                images = images.cuda()
                labels = labels.cuda()

            outputs = model(images)
            # top-1, top-5 ACC
            top1, top5 = accuracy(outputs, labels, topk=(1, 5))
            correct_top1 += top1
            correct_top5 += top5

    test_top1_acc = correct_top1.float() / len(test_loader.dataset)
    test_top5_acc = correct_top5.float() / len(test_loader.dataset)

    print()
    print(
        'Test set: Top1 Accuracy: {:.4f}, Top5 Accuracy: {:.4f}'.format(
            test_top1_acc,
            test_top5_acc,
    ))
