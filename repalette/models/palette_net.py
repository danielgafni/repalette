import torch
import torch.nn as nn

from repalette.models.nn import (
    ConvBlock,
    DeconvBlock,
    ResnetLayer,
    DenseLayer,
    DenseBottleneck,
    DenseBasicBlock,
    TransitionLayer,
    BasicBlock,
)


class PaletteNet(nn.Module):
    def __init__(self, mode='resnet'):
        super().__init__()
        if mode == 'resnet':
            self.feature_extractor = FeatureExtractor()
        else:
            self.feature_extractor = DenseFeatureExtractor(mode)
        self.recoloring_decoder = RecoloringDecoder()

    def forward(self, img, palette):
        luminance = img[:, 0:1, :, :]
        content_features = self.feature_extractor(img)
        recolored_img_ab = self.recoloring_decoder(content_features, palette, luminance)
        return recolored_img_ab


class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = ConvBlock(3, 64, padding_mode="replicate")
        self.pool = nn.MaxPool2d(2)

        self.res1 = ResnetLayer(BasicBlock, 64, 128)
        self.res2 = ResnetLayer(BasicBlock, 128, 256)
        self.res3 = ResnetLayer(BasicBlock, 256, 512)

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
        self.final_conv = ConvBlock(64 + 1, 2, activation="tanh", normalize=False)

    def forward(self, content_features, palette, luminance):
        device = next(self.parameters()).device
        c1, c2, c3, c4 = content_features

        batch_size, _, height, width = c1.size()
        palette_c1 = palette[:, :, None, None] * torch.ones(
            batch_size, 18, height, width, device=device
        )
        batch_size, _, height, width = c3.size()
        palette_c3 = palette[:, :, None, None] * torch.ones(
            batch_size, 18, height, width, device=device
        )
        batch_size, _, height, width = c4.size()
        palette_c4 = palette[:, :, None, None] * torch.ones(
            batch_size, 18, height, width, device=device
        )

        x = torch.cat([c1, palette_c1], dim=1)
        x = self.deconv1(x, c2.shape[-2:])

        x = torch.cat([x, c2], dim=1)
        x = self.deconv2(x, c3.shape[-2:])

        x = torch.cat([x, c3, palette_c3], dim=1)
        x = self.deconv3(x, c4.shape[-2:])

        x = torch.cat([x, c4, palette_c4], dim=1)
        x = self.deconv4(x, luminance.shape[-2:])

        x = torch.cat([luminance, x], dim=1)
        x = self.final_conv(x)

        return x


class DenseFeatureExtractor(nn.Module):
    def __init__(self, mode='dense-bc'):
        super().__init__()

        self.conv = ConvBlock(3, 64, padding_mode="replicate")
        self.pool = nn.MaxPool2d(2)

        if mode == 'dense-bc':
            self.dense1 = DenseLayer(DenseBottleneck, 6, 32, 64)
            self.trans1 = TransitionLayer(256, 128)
            self.dense2 = DenseLayer(DenseBottleneck, 12, 32, 128)
            self.trans2 = TransitionLayer(512, 256)
            self.dense3 = DenseLayer(DenseBottleneck, 24, 32, 256)
            self.trans3 = TransitionLayer(1024, 512)
        elif mode == 'dense-b':
            self.dense1 = DenseLayer(DenseBottleneck, 4, 16, 64)
            self.trans1 = TransitionLayer(128, 128)
            self.dense2 = DenseLayer(DenseBottleneck, 8, 16, 128)
            self.trans2 = TransitionLayer(256, 256)
            self.dense3 = DenseLayer(DenseBottleneck, 16, 16, 256)
            self.trans3 = TransitionLayer(512, 512)
        elif mode == 'dense':
            self.dense1 = DenseLayer(DenseBasicBlock, 4, 16, 64)
            self.trans1 = TransitionLayer(128, 128)
            self.dense2 = DenseLayer(DenseBasicBlock, 8, 16, 128)
            self.trans2 = TransitionLayer(256, 256)
            self.dense3 = DenseLayer(DenseBasicBlock, 16, 16, 256)
            self.trans3 = TransitionLayer(512, 512)

    def forward(self, x):
        x = self.conv(x)
        c4 = self.pool(x)
        c3 = self.trans1(self.dense1(c4))
        c2 = self.trans2(self.dense2(c3))
        c1 = self.trans3(self.dense3(c2))
        return c1, c2, c3, c4
