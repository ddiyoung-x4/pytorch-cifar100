import torch
import torch.nn as nn

class ConvNormAct(nn.Sequential):
    def __init__(self, in_ch, out_ch, kernel_size, norm_layer=nn.BatchNorm2d, stride=1, padding=0, groups=1, act=True):
        super(ConvNormAct, self).__init__(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias=False, groups=groups),
            norm_layer(out_ch),
            nn.ReLU(inplace=True) if act else nn.Identity()
        )

class CBAM(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()

        # channel attention module
        self.max_pool = nn.AdaptiveMaxPool2d((1,1))
        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(),
            nn.Linear(channel // reduction, channel)
        )
        self.sigmoid = nn.Sigmoid()

        # Spatial Attention Module
        self.spatial_attention = ConvNormAct(2, 1, kernel_size=7, padding=3)

    def forward(self, x):
        # channel attention
        max_out = self.max_pool(x)
        avg_out = self.avg_pool(x)
        max_out = self.fc(max_out.view(max_out.size(0), -1))
        avg_out = self.fc(avg_out.view(avg_out.size(0), -1))
        c_out = self.sigmoid(max_out + avg_out).unsqueeze(2).unsqueeze(3)
        c_out = x * c_out.expand_as(x)

        # spatial attention
        max_out = torch.max(c_out, 1).values.unsqueeze(1)
        avg_out = torch.mean(c_out, 1).unsqueeze(1)
        s_out = self.spatial_attention(torch.cat((max_out, avg_out), dim=1))
        s_out = self.sigmoid(s_out)

        return c_out * s_out.expand_as(c_out)