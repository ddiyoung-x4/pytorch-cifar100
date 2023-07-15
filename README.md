# pytorch-cifar100
PyTorch CIFAR-100 Implementation  

## Requirments
python3.8  
torch 1.11.0  
torchvision 0.12.0  

## Usage  
### 1. Datasets
You can download a python version of CIFAR-100 dataset by entering the official website(https://www.cs.toronto.edu/~kriz/cifar.html).  
And then you should change the path of cifar100-python folder in main.py.  

### 2. Train a Model
```
python main.py --model resnet50 --epochs 200 --batch-size 128 --optim sgd --momentum 0.9
(if you use CutMix) --beta 1.0 --cutmix-prob 0.5  
```
You can change the model ResNet 18, 34, 50, 101, 152, and sgd, adam, adamw for Optimizer.
