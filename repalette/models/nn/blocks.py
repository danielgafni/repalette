import torch.nn as nn
import torch

from repalette.models.nn.activations import activation_shortcuts


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
        blocks = [block(in_channels, out_channels, activation, stride)]
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


class DenseLayer(nn.Module):
    """DenseNet layer with dense connections."""

    def __init__(self, block, n_blocks, growth_rate, in_channels, activation='leaky_relu'):
        super().__init__()
        blocks = [block(in_channels + growth_rate * i, growth_rate, activation) for i in
                  range(n_blocks)]
        self.layer = nn.Sequential(*blocks)

    def forward(self, x):
        return self.layer(x)


class DenseBottleneck(nn.Module):
    """
    DenseNetB block aka bottleneck, consisting of two basic blocks
    with 1x1 and 3x3 convolutions respectively.
    """

    def __init__(self, in_channels, out_channels, activation="leaky_relu"):
        super().__init__()
        inter_channels = 4 * out_channels

        self.activ = activation_shortcuts[activation]

        self.norm1 = nn.InstanceNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, inter_channels, kernel_size=1, stride=1, padding=0)

        self.norm2 = nn.InstanceNorm2d(inter_channels)
        self.conv2 = nn.Conv2d(inter_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        out = self.conv1(self.activ(self.norm1(x)))
        out = self.conv2(self.activ(self.norm2(out)))
        return torch.cat([x, out], dim=-3)


class DenseBasicBlock(nn.Module):
    """DenseNet basic block consisting of batchnorm, activation and 3x3 convolution."""

    def __init__(self, in_channels, out_channels, activation="leaky_relu"):
        super().__init__()

        self.norm = nn.InstanceNorm2d(in_channels)
        self.activ = activation_shortcuts[activation]
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        out = self.conv(self.activ(self.norm(x)))
        return torch.cat([x, out], dim=-3)


class TransitionLayer(nn.Module):
    """Transition layer between adjacent densely connected dense blocks."""

    def __init__(self, in_channels, out_channels, activation="leaky_relu"):
        super().__init__()

        self.norm = nn.InstanceNorm2d(in_channels)
        self.activ = activation_shortcuts[activation]
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.pool = nn.AvgPool2d(2)

    def forward(self, x):
        x = self.norm(x)
        x = self.activ(x)
        x = self.conv(x)
        x = self.pool(x)
        return x


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
            ConvBlock(in_channels, out_channels, kernel_size, activation=activation),
            ConvBlock(out_channels, out_channels, kernel_size, activation=activation),
        )

    def forward(self, x, size=None):
        if size is not None:
            upsample = nn.Upsample(size=size, mode="bilinear", align_corners=True)
            x = upsample(x)
        return self.model(x)
