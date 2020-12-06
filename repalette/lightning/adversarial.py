import torch
import torch.nn as nn
import pytorch_lightning as pl

from repalette.models import Discriminator, PaletteNet
from repalette.utils.visualization import lab_batch_to_rgb_image_grid
from repalette.utils.normalize import Scaler
from repalette.constants import (
    DEFAULT_LR,
    DEFAULT_BETA_1,
    DEFAULT_BETA_2,
    DEFAULT_LAMBDA_MSE_LOSS,
)


class AdversarialTrainer(pl.LightningModule):
    """
    Wrapper for adversarial training of PaletteNet.
    """

    def __init__(
        self,
        palette_net: PaletteNet,
        train_dataloader,
        discriminator: Discriminator = None,
        val_dataloader=None,
        test_dataloader=None,
        lr=DEFAULT_LR,
        betas=(DEFAULT_BETA_1, DEFAULT_BETA_2),
        lambda_mse_loss=DEFAULT_LAMBDA_MSE_LOSS,
    ):
        super().__init__()
        self.generator = palette_net
        self.discriminator = (
            discriminator if discriminator is not None else Discriminator()
        )

        self.train_dataloader_ = train_dataloader
        self.val_dataloader_ = val_dataloader
        self.test_dataloader_ = test_dataloader
        self.loss_fn = nn.MSELoss()
        self.scaler = Scaler()
        self.hparams = {
            "lr": lr,
            "beta_1": betas[0],
            "beta_2": betas[1],
            "lambda_mse_loss": lambda_mse_loss,
        }

    def forward(self, img, palette):
        return self.generator(img, palette)

    def training_step(self, batch, batch_idx, optimizer_idx):
        (
            (source_img, _),
            (target_img, target_palette),
            (original_img, original_palette),
        ) = batch
        target_palette = nn.Flatten()(target_palette)
        original_palette = nn.Flatten()(original_palette)
        luminance = source_img[:, 0:1, :, :]
        recolored_img_ab = self.generator(source_img, target_palette)
        recolored_img = torch.cat([luminance, recolored_img_ab], dim=-3)
        mse_loss = self.loss_fn(recolored_img_ab, target_img[:, 1:, :, :])
        self.log("Train/Loss", mse_loss, prog_bar=True)
        if optimizer_idx == 0:
            real_prob_tt = self.discriminator(recolored_img, target_palette)
            adv_loss = -torch.mean(torch.log(real_prob_tt))
            return mse_loss * self.hparams["lambda_mse_loss"] + adv_loss
        if optimizer_idx == 1:
            fake_prob_tt = 1 - self.discriminator(recolored_img, target_palette)
            fake_prob_to = 1 - self.discriminator(recolored_img, original_palette)
            fake_prob_ot = 1 - self.discriminator(original_img, target_palette)
            real_prob_oo = self.discriminator(original_img, original_palette)
            adv_loss = -(
                torch.mean(torch.log(fake_prob_tt))
                + torch.mean(torch.log(fake_prob_to))
                + torch.mean(torch.log(fake_prob_ot))
                + torch.mean(torch.log(real_prob_oo))
            )
            return adv_loss

    def validation_step(self, batch, batch_idx):
        (original_img, _), (target_img, target_palette), _ = batch
        target_palette = nn.Flatten()(target_palette)
        recolored_img = self.generator(original_img, target_palette)
        loss = self.loss_fn(recolored_img, target_img[:, 1:, :, :])
        return loss

    def training_epoch_end(self, outputs):
        # OPTIONAL
        (original_img, _), (target_img, target_palette), _ = next(
            iter(self.train_dataloader())
        )

        original_img = original_img.to(self.device)
        target_img = target_img.to(self.device)
        target_palette = target_palette.to(self.device)

        with torch.no_grad():
            _target_palette = nn.Flatten()(target_palette)
            recolored_img = self.generator(original_img, _target_palette)

        original_luminance = original_img.clone()[:, 0:1, ...].to(self.device)
        recolored_img_with_luminance = torch.cat(
            (original_luminance, recolored_img), dim=1
        )

        self.scaler.to(self.device)

        original_img = self.scaler.inverse_transform(original_img)
        target_img = self.scaler.inverse_transform(target_img)
        target_palette = self.scaler.inverse_transform(target_palette)
        recolored_img_with_luminance = self.scaler.inverse_transform(
            recolored_img_with_luminance
        )

        original_grid = lab_batch_to_rgb_image_grid(original_img)
        target_grid = lab_batch_to_rgb_image_grid(target_img)

        target_palette_img = target_palette.view(-1, 3, 6, 1)
        target_palette_grid = lab_batch_to_rgb_image_grid(
            target_palette_img, pad_value=1.0, padding=1
        )

        recolored_grid = lab_batch_to_rgb_image_grid(recolored_img_with_luminance)

        self.logger.experiment.add_image(
            "Train/Original", original_grid, self.current_epoch
        )
        self.logger.experiment.add_image(
            "Train/Target", target_grid, self.current_epoch
        )
        self.logger.experiment.add_image(
            "Train/Target_Palette", target_palette_grid, self.current_epoch
        )
        self.logger.experiment.add_image(
            "Train/Recolored", recolored_grid, self.current_epoch
        )

    def validation_epoch_end(self, outputs):
        # OPTIONAL
        self.val_dataloader().shuffle(True)
        (original_img, _), (target_img, target_palette), _ = next(
            iter(self.val_dataloader())
        )
        self.val_dataloader().shuffle(False)

        original_img = original_img.to(self.device)
        target_img = target_img.to(self.device)
        target_palette = target_palette.to(self.device)

        with torch.no_grad():
            _target_palette = nn.Flatten()(target_palette)
            recolored_img = self.generator(original_img, _target_palette)

        self.logger.experiment.add_graph(self, (original_img, _target_palette))

        original_luminance = original_img.clone()[:, 0:1, ...].to(self.device)
        recolored_img_with_luminance = torch.cat(
            (original_luminance, recolored_img), dim=1
        )

        self.scaler.to(self.device)

        original_img = self.scaler.inverse_transform(original_img)
        target_img = self.scaler.inverse_transform(target_img)
        target_palette = self.scaler.inverse_transform(target_palette)
        recolored_img_with_luminance = self.scaler.inverse_transform(
            recolored_img_with_luminance
        )

        original_grid = lab_batch_to_rgb_image_grid(original_img)
        target_grid = lab_batch_to_rgb_image_grid(target_img)

        target_palette_img = target_palette.view(-1, 3, 6, 1)
        target_palette_grid = lab_batch_to_rgb_image_grid(
            target_palette_img, pad_value=1.0, padding=1
        )

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

        mean_val_loss = torch.stack(outputs).mean()
        self.log("Val/Loss", mean_val_loss)
        # self.logger.log_hyperparams(self.hparams)

    def train_dataloader(self):
        self.train_dataloader_.shuffle(True)  # train dataloader should be shuffled!
        return self.train_dataloader_

    def val_dataloader(self):
        return self.val_dataloader_

    def test_dataloader(self):
        return self.test_dataloader_

    def configure_optimizers(self):
        optimizer_G = torch.optim.Adam(
            self.generator.recoloring_decoder.parameters(),
            lr=self.hparams["lr"],
            betas=(self.hparams["beta_1"], self.hparams["beta_2"]),
        )
        optimizer_D = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=self.hparams["lr"],
            betas=(self.hparams["beta_1"], self.hparams["beta_2"]),
        )
        return [optimizer_G, optimizer_D]
