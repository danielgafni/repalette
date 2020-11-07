import torch
import torch.nn as nn
import pytorch_lightning as pl
import pandas as pd

from torch.utils.data import DataLoader

from collections import OrderedDict

from repalette.model_common.blocks import (
    ConvBlock,
    DeconvBlock,
    ResnetLayer,
    BasicBlock,
)
from repalette.constants import LR, BETAS
from repalette.utils import PairRecolorDataset


class PaletteNet(pl.LightningModule):
    def __init__(
        self,
        train_dataset,
        val_dataset=None,
        test_dataset=None,
        hparams={"lr": LR, "betas": BETAS, "batch_size": 32, "num_workers": 8},
    ):
        super().__init__()
        self.feature_extractor = FeatureExtractor()
        self.recoloring_decoder = RecoloringDecoder()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.loss_fn = nn.MSELoss()
        self.hparams = hparams

    def forward(self, img, palette):
        luminance = img[:, 0:1, :, :]
        content_features = self.feature_extractor(img)
        recolored_img_ab = self.recoloring_decoder(content_features, palette, luminance)
        return recolored_img_ab

    def training_step(self, batch, batch_idx):
        (original_img, _), (target_img, target_palette) = batch
        # print(original_img.shape)
        target_palette = nn.Flatten()(target_palette)
        recolored_img = self(original_img, target_palette)
        loss = self.loss_fn(recolored_img, target_img[:, 1:, :, :])
        self.log("Train/Loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        (original_img, _), (target_img, target_palette) = batch
        target_palette = nn.Flatten()(target_palette)
        recolored_img = self(original_img, target_palette)
        loss = self.loss_fn(recolored_img, target_img[:, 1:, :, :])
        self.log("Val/Loss", loss, prog_bar=True)
        return loss

    def train_dataloader(self):
        self.train_dataset.shuffle(True)  # train dataset should be shuffled!
        train_dataloader = DataLoader(
            self.train_dataset,
            shuffle=False,
            batch_size=self.hparams["batch_size"],
            num_workers=self.hparams["num_workers"],
        )
        return train_dataloader

    def val_dataloader(self):
        val_dataloader = DataLoader(
            self.val_dataset,
            shuffle=False,  # val dataset should not be shuffled!
            batch_size=self.hparams["batch_size"],
            num_workers=self.hparams["num_workers"],
        )
        return val_dataloader

    def test_dataloader(self):
        test_dataloader = DataLoader(
            self.test_dataset,
            shuffle=False,  # test dataset should not be shuffled!
            batch_size=self.hparams["batch_size"],
            num_workers=self.hparams["num_workers"],
        )
        return test_dataloader

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.hparams["lr"], betas=self.hparams["betas"]
        )
        return optimizer


class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = ConvBlock(3, 64)
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
        self.final_conv = ConvBlock(64 + 1, 2, activation="none")

    def forward(self, content_features, palette, luminance):
        c1, c2, c3, c4 = content_features

        batch_size, _, height, width = c1.size()
        palette_c1 = palette[:, :, None, None] * torch.ones(
            batch_size, 18, height, width
        )
        batch_size, _, height, width = c3.size()
        palette_c3 = palette[:, :, None, None] * torch.ones(
            batch_size, 18, height, width
        )
        batch_size, _, height, width = c4.size()
        palette_c4 = palette[:, :, None, None] * torch.ones(
            batch_size, 18, height, width
        )

        x = torch.cat([c1, palette_c1], dim=1)
        x = self.deconv1(x)

        x = torch.cat([x, c2], dim=1)
        x = self.deconv2(x)

        x = torch.cat([x, c3, palette_c3], dim=1)
        x = self.deconv3(x)

        x = torch.cat([x, c4, palette_c4], dim=1)
        x = self.deconv4(x)

        x = torch.cat([x, luminance], dim=1)
        x = self.final_conv(x)

        return x


class Discriminator(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.model = nn.Sequential(
            OrderedDict(
                [
                    ("conv1", ConvBlock(in_channels, 64, kernel_size=4, stride=2)),
                    ("conv2", ConvBlock(64, 64, kernel_size=4, stride=2)),
                    ("conv3", ConvBlock(64, 64, kernel_size=4, stride=2)),
                    ("conv4", ConvBlock(64, 64, kernel_size=4, stride=2)),
                    ("flatten", nn.Flatten()),
                    ("fc", nn.Linear(25600, 1)),
                    ("activ", nn.Sigmoid()),
                ]
            )
        )

    def forward(self, x):
        return self.model(x)
