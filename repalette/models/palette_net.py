import torch
import torch.nn as nn
import pytorch_lightning as pl

from collections import OrderedDict

from repalette.model_common.blocks import (
    ConvBlock,
    DeconvBlock,
    ResnetLayer,
    BasicBlock,
    FinalConvBlock
)
from repalette.utils.visualization import lab_batch_to_rgb_image_grid
from repalette.constants import DEFAULT_LR, DEFAULT_BETAS


class PaletteNet(pl.LightningModule):
    def __init__(
            self,
            train_dataloader,
            val_dataloader=None,
            test_dataloader=None,
            hparams=None,
    ):
        super().__init__()
        if hparams is None:
            hparams = {
                "lr": DEFAULT_LR,
                "betas": DEFAULT_BETAS,
                "batch_size": 32,
                "num_workers": 8,
            }
        self.feature_extractor = FeatureExtractor()
        self.recoloring_decoder = RecoloringDecoder()
        self.train_dataloader_ = train_dataloader
        self.val_dataloader_ = val_dataloader
        self.test_dataloader_ = test_dataloader
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

    def validation_epoch_end(self, outputs):
        # OPTIONAL
        self.val_dataloader().shuffle(True)
        (original_img, _), (target_img, target_palette) = next(
            iter(self.val_dataloader())
        )
        self.val_dataloader().shuffle(False)

        original_img = original_img.to(self.device)
        target_img = target_img.to(self.device)
        target_palette = target_palette.to(self.device)

        with torch.no_grad():
            target_palette = nn.Flatten()(target_palette)
            recolored_img = self(original_img, target_palette)

        original_grid = lab_batch_to_rgb_image_grid(original_img[0].unsqueeze(0))
        target_grid = lab_batch_to_rgb_image_grid(target_img)

        target_palette_img = target_palette.view(-1, 3, 6, 1)
        target_palette_grid = lab_batch_to_rgb_image_grid(
            target_palette_img, pad_value=1.0, padding=1
        )

        recolored_img_with_luminance = torch.cat((original_img[:, 0:1, :, :], recolored_img), dim=1)
        recolored_grid = lab_batch_to_rgb_image_grid(recolored_img_with_luminance)

        self.logger.experiment.add_image(
            "Val/Original", original_grid, self.current_epoch
        )
        self.logger.experiment.add_image("Val/Target", target_grid, self.current_epoch)
        self.logger.experiment.add_image(
            "Val/Target_Palette", target_palette_grid, self.current_epoch
        )
        self.logger.experiment.add_image(
            "Val/Recolored", recolored_grid, self.current_epoch
        )

        self.logger.experiment.add_graph(self, (original_img, target_palette))

    def train_dataloader(self):
        self.train_dataloader_.shuffle(True)  # train dataloader should be shuffled!
        return self.train_dataloader_

    def val_dataloader(self):
        return self.val_dataloader_

    def test_dataloader(self):
        return self.test_dataloader_

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.hparams["lr"], betas=self.hparams["betas"]
        )
        return optimizer


class FeatureExtractor(pl.LightningModule):
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


class RecoloringDecoder(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.deconv1 = DeconvBlock(512 + 18, 256)
        self.deconv2 = DeconvBlock(256 + 256, 128)
        self.deconv3 = DeconvBlock(128 + 128 + 18, 64)
        self.deconv4 = DeconvBlock(64 + 64 + 18, 64)
        self.final_conv = ConvBlock(64 + 1, 2, activation="none", normalize=False)

    def forward(self, content_features, palette, luminance):
        c1, c2, c3, c4 = content_features

        batch_size, _, height, width = c1.size()
        palette_c1 = palette[:, :, None, None] * torch.ones(
            batch_size, 18, height, width
        ).to(self.device)
        batch_size, _, height, width = c3.size()
        palette_c3 = palette[:, :, None, None] * torch.ones(
            batch_size, 18, height, width
        ).to(self.device)
        batch_size, _, height, width = c4.size()
        palette_c4 = palette[:, :, None, None] * torch.ones(
            batch_size, 18, height, width
        ).to(self.device)

        x = torch.cat([c1, palette_c1], dim=1)
        x = self.deconv1(x)

        x = torch.cat([x, c2], dim=1)
        x = self.deconv2(x)

        x = torch.cat([x, c3, palette_c3], dim=1)
        x = self.deconv3(x)

        x = torch.cat([x, c4, palette_c4], dim=1)
        x = self.deconv4(x)

        x = torch.cat([luminance, x], dim=1)
        x = self.final_conv(x)

        return x


class Discriminator(pl.LightningModule):
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
