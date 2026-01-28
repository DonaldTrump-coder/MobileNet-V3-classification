from models.modules import ChannelwiseSeparableConv, SeModule
import torch.nn as nn
from models.activations import HSigmoid, HSwish
import torch

class MobileNetV3Block(nn.Module):
    def __init__(self,
                 in_channels: int,
                 expand_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 nolinear: nn.Module,
                 stride: int = 1,
                 bias: bool = False,
                 se: bool = True,
                 reduction: int = 4
                 ):
        self.stride = stride
        self.inchannels = in_channels
        self.outchannels = out_channels
        self.se = se
        super(MobileNetV3Block, self).__init__()
        self.conv = ChannelwiseSeparableConv(in_channels, expand_channels, out_channels, kernel_size, nolinear, stride, bias)
        self.se = SeModule(out_channels, reduction)

    def forward(self, x):
        out = self.conv(x)
        if self.se:
            out = self.se(out)
        if self.stride == 1 and self.inchannels == self.outchannels:
            out = out + x
        return out
    
class small_MobileNetV3(nn.Module):
    def __init__(self, num_classes: int = 1000):
        super(small_MobileNetV3, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False), # 224x224x3 -> 112x112x16
            nn.BatchNorm2d(16),
            HSwish(),
            MobileNetV3Block(16, 16, 16, kernel_size=3, nolinear=nn.ReLU(inplace=True), stride=2, bias=False, se=True), # 112x112x16 -> 56x56x16
            MobileNetV3Block(16, 72, 24, kernel_size=3, nolinear=nn.ReLU(inplace=True), stride=2, bias=False, se=False), # 56x56x16 -> 28x28x24
            MobileNetV3Block(24, 88, 24, kernel_size=3, nolinear=nn.ReLU(inplace=True), stride=1, bias=False, se=False), # 28x28x24 -> 28x28x24
            MobileNetV3Block(24, 96, 40, kernel_size=5, nolinear=HSwish(), stride=2, bias=False, se=True), # 28x28x24 -> 14x14x40
            MobileNetV3Block(40, 240, 40, kernel_size=5, nolinear=HSwish(), stride=1, bias=False, se=True), # 14x14x40 -> 14x14x40
            MobileNetV3Block(40, 240, 40, kernel_size=5, nolinear=HSwish(), stride=1, bias=False, se=True), # 14x14x40 -> 14x14x40
            MobileNetV3Block(40, 120, 48, kernel_size=5, nolinear=HSwish(), stride=1, bias=False, se=True), # 14x14x40 -> 14x14x48
            MobileNetV3Block(48, 144, 48, kernel_size=5, nolinear=HSwish(), stride=1, bias=False, se=True), # 14x14x48 -> 14x14x48
            MobileNetV3Block(48, 288, 96, kernel_size=5, nolinear=HSwish(), stride=2, bias=False, se=True), # 14x14x48 -> 7x7x96
            MobileNetV3Block(96, 576, 96, kernel_size=5, nolinear=HSwish(), stride=1, bias=False, se=True), # 7x7x96 -> 7x7x96
            MobileNetV3Block(96, 576, 96, kernel_size=5, nolinear=HSwish(), stride=1, bias=False, se=True), # 7x7x96 -> 7x7x96
            nn.Conv2d(96, 576, kernel_size=1, stride=1, padding=0, bias=False), # 7x7x96 -> 7x7x576
            nn.BatchNorm2d(576),
            HSwish(),
            nn.AdaptiveAvgPool2d(1), # 7x7x576 -> 1x1x576
            nn.Conv2d(576, 1024, kernel_size=1, stride=1, padding=0, bias=False), # 1x1x576 -> 1x1x1024
            HSwish()
        )
        self.output_head = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.model(x)
        x = torch.flatten(x, 1)
        x = self.output_head(x)
        return x
    
class large_MobileNetV3(nn.Module):
    def __init__(self, num_classes: int = 1000):
        super(large_MobileNetV3, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False), # 224x224x3 -> 112x112x16
            nn.BatchNorm2d(16),
            HSwish(),
            MobileNetV3Block(16, 16, 16, kernel_size=3, nolinear=nn.ReLU(inplace=True), stride=1, bias=False, se=False), # 112x112x16 -> 112x112x16
            MobileNetV3Block(16, 64, 24, kernel_size=3, nolinear=nn.ReLU(inplace=True), stride=2, bias=False, se=False), # 112x112x16 -> 56x56x24
            MobileNetV3Block(24, 72, 24, kernel_size=3, nolinear=nn.ReLU(inplace=True), stride=1, bias=False, se=False), # 56x56x24 -> 56x56x24
            MobileNetV3Block(24, 72, 40, kernel_size=5, nolinear=nn.ReLU(inplace=True), stride=2, bias=False, se=True), # 56x56x24 -> 28x28x40
            MobileNetV3Block(40, 120, 40, kernel_size=5, nolinear=nn.ReLU(inplace=True), stride=1, bias=False, se=True), # 28x28x40 -> 28x28x40
            MobileNetV3Block(40, 120, 40, kernel_size=5, nolinear=nn.ReLU(inplace=True), stride=1, bias=False, se=True), # 28x28x40 -> 28x28x40
            MobileNetV3Block(40, 240, 80, kernel_size=3, nolinear=HSwish(), stride=2, bias=False, se=False), # 28x28x40 -> 14x14x80
            MobileNetV3Block(80, 200, 80, kernel_size=3, nolinear=HSwish(), stride=1, bias=False, se=False), # 14x14x80 -> 14x14x80
            MobileNetV3Block(80, 184, 80, kernel_size=3, nolinear=HSwish(), stride=1, bias=False, se=False), # 14x14x80 -> 14x14x80
            MobileNetV3Block(80, 184, 80, kernel_size=3, nolinear=HSwish(), stride=1, bias=False, se=False), # 14x14x80 -> 14x14x80
            MobileNetV3Block(80, 480, 112, kernel_size=3, nolinear=HSwish(), stride=1, bias=False, se=True), # 14x14x80 -> 14x14x112
            MobileNetV3Block(112, 672, 112, kernel_size=3, nolinear=HSwish(), stride=1, bias=False, se=True), # 14x14x112 -> 14x14x112
            MobileNetV3Block(112, 672, 160, kernel_size=5, nolinear=HSwish(), stride=2, bias=False, se=True), # 14x14x112 -> 7x7x160
            MobileNetV3Block(160, 960, 160, kernel_size=5, nolinear=HSwish(), stride=1, bias=False, se=True), # 7x7x160 -> 7x7x160
            MobileNetV3Block(160, 960, 160, kernel_size=5, nolinear=HSwish(), stride=1, bias=False, se=True), # 7x7x160 -> 7x7x160
            nn.Conv2d(160, 960, kernel_size=1, stride=1, padding=0, bias=False), # 7x7x160 -> 7x7x960
            nn.BatchNorm2d(960),
            HSwish(),
            nn.AdaptiveAvgPool2d(1), # 7x7x960 -> 1x1x960
            nn.Conv2d(960, 1024, kernel_size=1, stride=1, padding=0, bias=False), # 1x1x960 -> 1x1x1024
            HSwish()
        )
        self.output_head = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.model(x)
        x = torch.flatten(x, 1)
        x = self.output_head(x)
        return x