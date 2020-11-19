# import torch
import torch.nn as nn

from repalette.model_common.activations import activation_shortcuts


class ResnetLayer(nn.Module):
    """Resnet layer consisting of several residual blocks."""

    def __init__(
        self,
        block,
        in_channels,
        out_channels,
        n_blocks=2,
        activation="leaky_relu",
        stride=2,
    ):
        super().__init__()
        blocks = []
        blocks.append(block(in_channels, out_channels, activation, stride))
        for _ in range(1, n_blocks):
            blocks.append(block(out_channels, out_channels, activation))

        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        return self.model(x)


class ResidualBlock(nn.Module):
    """Parent class for residual blocks in ResNet. Inherit from it to define custom architecture."""

    def __init__(self):
        super().__init__()

        self.residual = nn.Identity()
        self.shortcut = nn.Identity()
        self.activation = nn.Identity()

    def forward(self, x):
        residual = self.residual(x)
        shortcut = self.shortcut(x)
        return self.activation(residual + shortcut)


class BasicBlock(ResidualBlock):
    """Basic block for ResNet consisting of 2 convolutional layers."""

    def __init__(self, in_channels, out_channels, activation="leaky_relu", stride=1):
        super().__init__()

        self.residual = nn.Sequential(
            ConvBlock(in_channels, out_channels, stride=stride, activation=activation),
            ConvBlock(out_channels, out_channels, activation="none"),
        )
        if stride != 1 or in_channels != out_channels:
            self.shortcut = ConvBlock(
                in_channels,
                out_channels,
                kernel_size=1,
                padding=0,
                stride=stride,
                activation="none",
            )
        self.activation = activation_shortcuts[activation]


class ConvBlock(nn.Module):
    """Convolution layer followed by instance normalization and activation function."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        dilation=1,
        activation="leaky_relu",
        normalize=True,
        padding_mode="zeros",
    ):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            padding_mode=padding_mode,
        )
        if normalize:
            self.norm = nn.InstanceNorm2d(out_channels)
        else:
            self.norm = nn.Identity()
        self.activ = activation_shortcuts[activation]

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.activ(x)
        return x


class DeconvBlock(nn.Module):
    """Upsampling block consisting of 2 convolutional blocks."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        activation="leaky_relu",
    ):
        super().__init__()
        self.model = nn.Sequential(
            ConvBlock(in_channels, out_channels, kernel_size),
            ConvBlock(out_channels, out_channels, kernel_size),
        )

    def forward(self, x, size=None):
        if size is not None:
            upsample = nn.Upsample(size=size, mode="bilinear", align_corners=True)
            x = upsample(x)
        return self.model(x)
