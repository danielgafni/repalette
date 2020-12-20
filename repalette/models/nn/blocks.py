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

    def __init__(self, block, growth_rate, n_blocks):
        super().__init__()
        self.blocks = [block(growth_rate * (n_prev_layers + 1), growth_rate) for n_prev_layers in
                       range(n_blocks)]

    def forward(self, x):
        for block in self.blocks[-1]:
            layer_out = block(x)
            x = torch.cat((x, layer_out), dim=-3)
        x = self.blocks[-1](x)
        return x


class DenseBottleneck(nn.Module):
    """
    DenseNetB block aka bottleneck, consisting of two basic blocks
    with 1x1 and 3x3 convolutions respectively.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.first_block = DenseBasicBlock(in_channels, 4 * out_channels, 1)
        self.second_block = DenseBasicBlock(4 * out_channels, out_channels)

    def forward(self, x):
        x = self.first_block(x)
        x = self.second_block(x)
        return x


class DenseBasicBlock(nn.Module):
    """DenseNet basic block consisting of batchnorm, activation and 3x3 convolution."""

    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            dilation=1,
            activation="leaky_relu",
            padding_mode="zeros",
    ):
        super().__init__()

        self.norm = nn.InstanceNorm2d(in_channels)
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            padding_mode=padding_mode,
        )
        self.activ = activation_shortcuts[activation]

    def forward(self, x):
        x = self.norm(x)
        x = self.activ(x)
        x = self.conv(x)
        return x


class TransitionLayer(nn.Module):
    """Transition layer between adjacent densely connected dense blocks."""

    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=1,
            dilation=1,
            padding_mode="zeros",
    ):
        super().__init__()
        self.norm = nn.InstanceNorm2d(in_channels)
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            padding_mode=padding_mode,
        )
        self.pool = nn.AvgPool2d(2)

    def forward(self, x):
        x = self.norm(x)
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
