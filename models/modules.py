import torch.nn as nn
from models.activations import HSigmoid

class ChannelwiseSeparableConv(nn.Module):
    def __init__(self,
                 in_channels: int, # input channels
                 expand_channels: int, # expand the input channels
                 out_channels: int, # 1x1 conv output channels
                 kernel_size: int, # conv kernel size of channelwise conv
                 nolinear: nn.Module, # non-linear activation function (HSigmoid or HSwish)
                 stride: int = 1,
                 bias: bool = False):
        super(ChannelwiseSeparableConv, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, expand_channels, kernel_size=1, stride=1, padding=0, bias=bias)
        self.bn1 = nn.BatchNorm2d(expand_channels)
        self.nolinear1 = nolinear

        self.conv2 = nn.Conv2d(expand_channels, expand_channels, kernel_size=kernel_size, stride=stride, padding = kernel_size // 2, groups=expand_channels, bias=bias) # SeparableConv
        self.bn2 = nn.BatchNorm2d(expand_channels)
        self.nolinear2 = nolinear

        self.conv3 = nn.Conv2d(expand_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=bias)
        self.bn3 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.nolinear1(self.bn1(self.conv1(x)))
        x = self.nolinear2(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        return x

class SeModule(nn.Module): # weighted channel attention module
    def __init__(self, in_channels: int, reduction: int = 4):
        super(SeModule, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), # pooling to 1x1
            nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(in_channels // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(in_channels),
            HSigmoid()
        )

    def forward(self, x):
        return x * self.se(x)