import torch
import pytorch_lightning as pl
from pytorch_lightning.metrics.regression import MeanSquaredError
from torch import nn as nn

from repalette.constants import (
    DEFAULT_ADVERSARIAL_LR,
    DEFAULT_ADVERSARIAL_BETA_1,
    DEFAULT_ADVERSARIAL_BETA_2,
    DEFAULT_ADVERSARIAL_LAMBDA_MSE_LOSS,
)
from repalette.models import PaletteNet, Discriminator
from repalette.utils.transforms import Scaler


class PreTrainSystem(pl.LightningModule):
    """
    Wrapper for pre-training of PaletteNet.
    """

    def __init__(
        self,
        learning_rate,
        beta_1,
        beta_2,
        weight_decay,
        optimizer,
        batch_size,
        multiplier,
        scheduler_patience,
    ):
        super().__init__()

        self.save_hyperparameters()  # `self.__init__` arguments are saved to `self.hparams`

        self.generator = PaletteNet()
        # self.MSE = MeanSquaredError()
        self.MSE = torch.nn.MSELoss()
        self.scaler = Scaler()

    def forward(self, img, palette):
        return self.generator(img, palette)

    def training_step(self, batch, batch_idx):
        (source_img, _), (target_img, target_palette) = batch
        target_palette = nn.Flatten()(target_palette)
        recolored_img_ab = self.generator(source_img, target_palette)
        loss = self.MSE(recolored_img_ab, target_img[:, 1:, :, :])
        self.log("Train/loss_step", loss)

        return loss

    def training_epoch_end(self, outputs):
        # log training loss
        self.log("Train/loss_epoch", torch.stack([output["loss"] for output in outputs]).mean())

    def validation_step(self, batch, batch_idx):
        (source_img, _), (target_img, target_palette) = batch
        target_palette = nn.Flatten()(target_palette)
        recolored_img_ab = self.generator(source_img, target_palette)
        loss = self.MSE(recolored_img_ab, target_img[:, 1:, :, :])

        return loss

    def validation_epoch_end(self, outputs):
        # log validation loss
        # self.log("Val/loss_epoch", self.MSE)
        self.log("Val/loss_epoch", torch.stack(outputs).mean())
        self.logger.log_hyperparams(self.hparams)

    def test_step(self, batch, batch_idx):
        (source_img, _), (target_img, target_palette) = batch
        target_palette = nn.Flatten()(target_palette)
        recolored_img_ab = self.generator(source_img, target_palette)
        loss = self.MSE(recolored_img_ab, target_img[:, 1:, :, :])

        return loss

    def test_epoch_end(self, outputs):
        # log test loss
        loss_epoch = torch.stack(outputs).mean()
        self.log("Test/loss_epoch", loss_epoch)
        self.logger.log_hyperparams(self.hparams, loss_epoch)

    def configure_optimizers(self):
        # which is better? adam or adamw?
        if self.hparams.optimizer == "adam":
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.hparams.learning_rate,
                betas=(self.hparams.beta_1, self.hparams.beta_2),
                weight_decay=self.hparams.weight_decay,
            )
        elif self.hparams.optimizer == "adamw":
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.hparams.learning_rate,
                betas=(self.hparams.beta_1, self.hparams.beta_2),
                weight_decay=self.hparams.weight_decay,
            )
        else:
            raise NotImplementedError(
                f"Optimizer {self.hparams.optimizer} is not implemented"
            )

        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer, mode="min", patience=self.hparams.scheduler_patience
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler,
            "monitor": "Val/loss_epoch",
        }

    @property
    def example_input_array(self):
        (source_img, _), (target_img, target_palette) = next(
            iter(self.val_dataloader())
        )
        return source_img[0:1, ...], nn.Flatten()(target_palette[0:1, ...])


class AdversarialSystem(pl.LightningModule):
    """
    Wrapper for adversarial training of PaletteNet.
    """

    # TODO: refactor

    def __init__(
        self,
        palette_net: PaletteNet,
        train_dataloader,
        discriminator: Discriminator = None,
        val_dataloader=None,
        test_dataloader=None,
        lr=DEFAULT_ADVERSARIAL_LR,
        betas=(DEFAULT_ADVERSARIAL_BETA_1, DEFAULT_ADVERSARIAL_BETA_2),
        lambda_mse_loss=DEFAULT_ADVERSARIAL_LAMBDA_MSE_LOSS,
    ):
        super().__init__()
        self.generator = palette_net
        self.discriminator = (
            discriminator if discriminator is not None else Discriminator()
        )

        self.train_dataloader_ = train_dataloader
        self.val_dataloader_ = val_dataloader
        self.test_dataloader_ = test_dataloader
        self.MSE = MeanSquaredError()
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
        mse_loss = self.MSE(recolored_img_ab, target_img[:, 1:, :, :])
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
        loss = self.MSE(recolored_img, target_img[:, 1:, :, :])
        return loss

    def training_epoch_end(self, outputs):
        pass

    def validation_epoch_end(self, outputs):
        mean_val_loss = torch.stack(outputs).mean()
        self.log("Val/Loss", mean_val_loss)
        # self.logger.log_hyperparams(self.hparams)

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
