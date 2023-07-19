import torch.nn as nn

class SE_Layer(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()

        self.global_avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.global_avg_pool(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        out = out.view(x.size(0), x.size(1), 1, 1)

        return x * out.expand_as(x)

class SE_BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        self.se_layer = SE_Layer(out_channels)

    def forward(self, x):
        out = nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))
        out = self.se_layer(out)

        return out

class SE_BottleNeck(nn.Module):
    expansion = 4
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels * SE_BottleNeck.expansion, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels * SE_BottleNeck.expansion),
        )
        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels * SE_BottleNeck.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * SE_BottleNeck.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * SE_BottleNeck.expansion)
            )
        self.se_layer = SE_Layer(out_channels * SE_BottleNeck.expansion)

    def forward(self, x):
        out = nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))
        out = self.se_layer(out)

        return out

class SE_ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=100):
        super().__init__()

        self.in_channels = 64

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        # self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv2_x = self.make_layer(64, block, num_blocks[0], 1)
        self.conv3_x = self.make_layer(128, block, num_blocks[1], 2)
        self.conv4_x = self.make_layer(256, block, num_blocks[2], 2)
        self.conv5_x = self.make_layer(512, block, num_blocks[3], 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512*block.expansion, num_classes)
    def make_layer(self, out_channels, block, num_block, stride):
        layers = []
        strides = [stride] + [1] * (num_block-1)

        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2_x(out)
        out = self.conv3_x(out)
        out = self.conv4_x(out)
        out = self.conv5_x(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out



def se_resnet18():
    """ return a ResNet 18 object
    """
    return SE_ResNet(SE_BasicBlock, [2, 2, 2, 2], num_classes=100)

def se_resnet34():
    """ return a ResNet 34 object
    """
    return SE_ResNet(SE_BasicBlock, [3, 4, 6, 3], num_classes=100)

def se_resnet50():
    """ return a ResNet 50 object
    """
    return SE_ResNet(SE_BottleNeck, [3, 4, 6, 3], num_classes=100)

def se_resnet101():
    """ return a ResNet 101 object
    """
    return SE_ResNet(SE_BottleNeck, [3, 4, 23, 3], num_classes=100)

def se_resnet152():
    """ return a ResNet 152 object
    """
    return SE_ResNet(SE_BottleNeck, [3, 8, 36, 3], num_classes=100)

if __name__ == "__main__":
    import torchsummary

    model = se_resnet50().cuda()
    torchsummary.summary(model, (3, 32, 32))