import torch.nn as nn

class ChannelwiseSeparableConv(nn.Module):
    def __init__(self, in_channels: int, expand_channels: int, out_channels: int, kernel_size: int, nolinear: nn.Module, stride: int = 1, bias: bool = False):

        self.conv1 = nn.Conv2d(in_channels, expand_channels, kernel_size=1, stride=1, padding=0, bias=bias)
        self.bn1 = nn.BatchNorm2d(expand_channels)
        self.nolinear1 = nolinear

        self.conv2 = nn.Conv2d(expand_channels, expand_channels, kernel_size=kernel_size, stride=stride, padding= kernel_size // 2, groups=expand_channels, bias=bias) # SeparableConv
        self.bn2 = nn.BatchNorm2d(expand_channels)
        self.nolinear2 = nolinear

        self.conv3 = nn.Conv2d(expand_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=bias)
        self.bn3 = nn.BatchNorm2d(out_channels)

class SeModule(nn.Module):
    def __init__(self, in_channels: int, reduction: int = 4):
        super(SeModule, self).__init__()