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

### 3. Test the Model
```
python test.py --model resnet50 --weights path_resnet50_weights_file --batch-size 4 --gpu True  
```
    
## Training Details  
#### 1. The training hyper-parameters
I trained ResNet50 model for 200 epochs and acheived top1 accruacy - 83.21%, top5 accuracy 96.21% of test dataset by using input image size (32x32x3) and ResNet which 1st layer is replaced with 3x3 convolution layer(you can check the code, resnet.py) because of small image size. The customized ResNet50 has 23.7M parameters and 199G FLOPS.  
  
#### 2. Experiment results
|  Dataset  |   Network   | Img size | Params | FLOPS | Top-1 Acc | Top-5 Acc | Epohcs | 
| :-------: | :-------: | ------- | :-------: | :-------: | :-------: | :-------: | :-------: |
| cifar100 | resnet50 | 32x32x3 | 23.7M | 199G | 83.21 | 96.21 | 200 |

##### eval-top1-acc graph
<img width="50%" src="https://github.com/ddiyoung-x4/pytorch-cifar100/assets/69739208/d69033c5-8665-430c-8037-9e0fc4b90aea"/>

##### eval-top5-acc graph  
<img width="50%" src="https://github.com/ddiyoung-x4/pytorch-cifar100/assets/69739208/5ea2f515-f619-4abe-9c42-ceb1fd3c8ff5"/>  
  
#### 3. Novel argument for fast network convergences  
I argue that the CutMix, AutoAugment are the key factors for fast newtork convergence. In the experiment, only CutMix strategy shows +3.09% accruacy improvement and only AutoAugment shows +1.19% accuracy improvement. Using both CutMix and AutoAugment makes the model improve +4.97% accuracy.
<img width="70%" src="https://github.com/ddiyoung-x4/pytorch-cifar100/assets/69739208/041ed1f0-dcf1-4cf2-bd32-5b8d73cce055"/>  

