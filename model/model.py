import torch
import torch.nn as nn

from collections import OrderedDict

from blocks import ConvBlock, DeconvBlock, ResnetLayer


class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = ConvBlock(3, 64)
        self.pool = nn.MaxPool2d(2)

        self.res1 = ResnetLayer(64, 128)
        self.res2 = ResnetLayer(128, 256)
        self.res3 = ResnetLayer(256, 512)

    def forward(self, x):
        x = self.conv(x)
        c4 = self.pool(x)
        c3 = self.res1(c4)
        c2 = self.res2(c3)
        c1 = self.res3(c2)
        return c1, c2, c3, c4


class RecoloringDecoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.deconv1 = DeconvBlock(512 + 18, 256)
        self.deconv2 = DeconvBlock(256 + 256, 128)
        self.deconv3 = DeconvBlock(128 + 128 + 18, 64)
        self.deconv4 = DeconvBlock(64 + 64 + 18, 64)
        self.final_conv = ConvBlock(64 + 1, 2, activation=None)

    def forward(self, content_features, palette, luminance):
        c1, c2, c3, c4 = content_features
        batch_size, _, height, width = c1.size()
        palette = palette[None, :, None, None] * torch.ones(batch_size, 18, height, width)

        x = torch.cat([c1, palette], dim=1)
        x = self.deconv1(x)

        x = torch.cat([x, c2], dim=1)
        x = self.deconv2(x)

        x = torch.cat([x, c3, palette], dim=1)
        x = self.deconv3(x)

        x = torch.cat([x, c4, palette], dim=1)
        x = self.deconv4(x)

        x = torch.cat([x, luminance], dim=1)
        x = self.final_conv(x)

        x = torch.cat([x, luminance], dim=1)
        return x


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
            ('activ', nn.Sigmoid())
        ]))

    def forward(self, x):
        return self.model(x)
