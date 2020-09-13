import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict

from activations import activation_shortcuts


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=0, dilation=1, activation='leaky_relu'):
        super().__init__()
        self.block = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation)),
            ('norm', nn.InstanceNorm2d(out_channels))
            ('activ', activation_shortcuts[activation])
        ]))

    def forward(self, x):
        return self.block(x)


class ResBlock(nn.Module):
    pass


class FeatureExtractor(nn.Module):
    pass


class RecoloringDecoder(nn.Module):
    pass


class Discriminator(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.model = nn.Sequential(OrderedDict([
            ('conv1', ConvBlock(in_channels, 64, kernel_size=4, stride=2)),
            ('conv2', ConvBlock(64, 64, kernel_size=4, stride=2)),
            ('conv3', ConvBlock(64, 64, kernel_size=4, stride=2)),
            ('conv4', ConvBlock(64, 64, kernel_size=4, stride=2)),
            ('flatten', nn.Flatten()),
            ('fc', nn.Linear(25600, 1)),
        ]))

    def forward(self, x):
        return F.sigmoid(self.model(x))
