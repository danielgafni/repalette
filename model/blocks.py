# import torch
import torch.nn as nn
# import torch.nn.functional as F

from collections import OrderedDict

from activations import activation_shortcuts


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, activation='leaky_relu'):
        super().__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation)
        self.norm = nn.InstanceNorm2d(out_channels)
        self.activ = activation_shortcuts[activation]

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.activ(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self):
        self.residual = nn.Identity()
        self.shortcut = nn.Identity()
        self.activation = nn.Identity()

    def forward(self, x):
        residual = self.residual(x)
        shortcut = self.shortcut(x)
        return self.activation(residual + shortcut)


class BasicBlock(ResidualBlock):
    def __init__(self, in_channels, out_channels, activation='leaky_relu', stride=1):
        super().__init__()

        self.residual = nn.Sequential(
            ConvBlock(in_channels, out_channels, stride=stride),
            ConvBlock(out_channels, out_channels, activation=None),
        )
        if stride != 1 or in_channels != out_channels:
            self.shortcut = ConvBlock(in_channels, out_channels, kernel_size=1, stride=stride, activation=None)
        self.activation = activation_shortcuts[activation]


class ResnetLayer(nn.Module):
    def __init__(self, block, in_channels, out_channels, n_blocks=2, activation='leaky_relu', stride=2):
        blocks = []
        blocks.append(block(in_channels, out_channels, activation, stride))
        for _ in range(1, n_blocks):
            blocks.append(block(out_channels, out_channels, activation))

        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        return self.model(x)
