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

        # manually configure hyperparams in case we add some non-hyperparams arguments in the future
        # they are saved to `self.hparams`
        self.save_hyperparameters(
            "learning_rate",
            "beta_1",
            "beta_2",
            "weight_decay",
            "optimizer",
            "batch_size",
            "multiplier",
            "scheduler_patience",
        )

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
        self.log("Val/loss_step", loss)

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
        self.log("Test/loss_step", loss)

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
            raise NotImplementedError(f"Optimizer {self.hparams.optimizer} is not implemented")

        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer, mode="min", patience=self.hparams.scheduler_patience
        )

        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler, "monitor": "Val/loss_epoch"}

    @property
    def example_input_array(self):
        (source_img, _), (target_img, target_palette) = next(iter(self.val_dataloader()))
        return source_img[0:1, ...], nn.Flatten()(target_palette[0:1, ...])


class AdversarialSystem(pl.LightningModule):
    """
    Wrapper for adversarial training of PaletteNet.
    """

    # TODO: refactor

    def __init__(
        self,
        generator,
        discriminator=None,
        lambda_mse_loss=DEFAULT_ADVERSARIAL_LAMBDA_MSE_LOSS,
        learning_rate=DEFAULT_ADVERSARIAL_LR,
        beta_1=DEFAULT_ADVERSARIAL_BETA_1,
        beta_2=DEFAULT_ADVERSARIAL_BETA_2,
        weight_decay=0.01,
        optimizer="adam",
        batch_size=8,
        multiplier=16,
    ):
        super().__init__()

        # manually configure hyperparams in case we add some non-hyperparams arguments in the future
        # they are saved to `self.hparams`
        self.save_hyperparameters(
            "lambda_mse_loss",
            "learning_rate",
            "beta_1",
            "beta_2",
            "weight_decay",
            "optimizer",
            "batch_size",
            "multiplier",
            "scheduler_patience",
        )

        self.generator = None
        self.discriminator = Discriminator()

        self.MSE = MeanSquaredError()
        self.scaler = Scaler()

    def set_generator(self, generator):
        self.generator = generator

    def set_discriminator(self, discriminator):
        self.discriminator = discriminator

    def forward(self, img, palette):
        return self.generator(img, palette)

    def training_step(self, batch, batch_idx, optimizer_idx):
        ((source_img, _), (target_img, target_palette), (original_img, original_palette)) = batch
        target_palette = nn.Flatten()(target_palette)
        original_palette = nn.Flatten()(original_palette)
        luminance = source_img[:, 0:1, :, :]
        recolored_img_ab = self.generator(source_img, target_palette)
        recolored_img = torch.cat([luminance, recolored_img_ab], dim=-3)
        mse_loss = self.MSE(recolored_img_ab, target_img[:, 1:, :, :])

        adv_loss = 0
        if optimizer_idx == 0:
            real_prob_tt = self.discriminator(recolored_img, target_palette)
            adv_loss = -torch.mean(torch.log(real_prob_tt))
            return mse_loss * self.hparams.lambda_mse_loss + adv_loss
        elif optimizer_idx == 1:
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
        else:
            pass

        self.log("Train/mse_loss_step", mse_loss)
        self.log("Train/adv_loss_step", adv_loss)

        return mse_loss, adv_loss

    def training_epoch_end(self, outputs):
        # log training loss
        self.log(
            "Train/mse_loss_epoch", torch.stack([output["mse_loss"] for output in outputs]).mean()
        )
        self.log(
            "Train/adv_loss_epoch", torch.stack([output["adv_loss"] for output in outputs]).mean()
        )

    def validation_step(self, batch, batch_idx, optimizer_idx):
        ((source_img, _), (target_img, target_palette), (original_img, original_palette)) = batch
        target_palette = nn.Flatten()(target_palette)
        original_palette = nn.Flatten()(original_palette)
        luminance = source_img[:, 0:1, :, :]
        recolored_img_ab = self.generator(source_img, target_palette)
        recolored_img = torch.cat([luminance, recolored_img_ab], dim=-3)
        mse_loss = self.MSE(recolored_img_ab, target_img[:, 1:, :, :])

        adv_loss = 0
        if optimizer_idx == 0:
            real_prob_tt = self.discriminator(recolored_img, target_palette)
            adv_loss = -torch.mean(torch.log(real_prob_tt))
            return mse_loss * self.hparams.lambda_mse_loss + adv_loss
        elif optimizer_idx == 1:
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
        else:
            pass

        return mse_loss, adv_loss

    def validation_epoch_end(self, outputs):
        # log training loss
        self.log(
            "Val/mse_loss_epoch", torch.stack([output["mse_loss"] for output in outputs]).mean()
        )
        self.log(
            "Val/adv_loss_epoch", torch.stack([output["adv_loss"] for output in outputs]).mean()
        )

    def test_step(self, batch, batch_idx, optimizer_idx):
        ((source_img, _), (target_img, target_palette), (original_img, original_palette)) = batch
        target_palette = nn.Flatten()(target_palette)
        original_palette = nn.Flatten()(original_palette)
        luminance = source_img[:, 0:1, :, :]
        recolored_img_ab = self.generator(source_img, target_palette)
        recolored_img = torch.cat([luminance, recolored_img_ab], dim=-3)
        mse_loss = self.MSE(recolored_img_ab, target_img[:, 1:, :, :])

        adv_loss = 0
        if optimizer_idx == 0:
            real_prob_tt = self.discriminator(recolored_img, target_palette)
            adv_loss = -torch.mean(torch.log(real_prob_tt))
            return mse_loss * self.hparams.lambda_mse_loss + adv_loss
        elif optimizer_idx == 1:
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
        else:
            pass

        return mse_loss, adv_loss

    def test_epoch_end(self, outputs):
        # log training loss
        self.log(
            "Test/mse_loss_epoch", torch.stack([output["mse_loss"] for output in outputs]).mean()
        )
        self.log(
            "Test/adv_loss_epoch", torch.stack([output["adv_loss"] for output in outputs]).mean()
        )

    def configure_optimizers(self):
        if self.hparams.optimizer == "adam":
            optimizer_G = torch.optim.Adam(
                self.generator.recoloring_decoder.parameters(),
                lr=self.learning_rate,
                betas=(self.hparams.beta_1, self.beta_2),
                weight_decay=self.hparams.weight_decay,
            )
            optimizer_D = torch.optim.Adam(
                self.discriminator.parameters(),
                lr=self.learning_rate,
                betas=(self.beta_1, self.beta_2),
                weight_decay=self.hparams.weight_decay,
            )

        elif self.hparams.optimizer == "adamw":
            optimizer_G = torch.optim.AdamW(
                self.generator.recoloring_decoder.parameters(),
                lr=self.learning_rate,
                betas=(self.hparams.beta_1, self.beta_2),
                weight_decay=self.hparams.weight_decay,
            )
            optimizer_D = torch.optim.AdamW(
                self.discriminator.parameters(),
                lr=self.learning_rate,
                betas=(self.beta_1, self.beta_2),
                weight_decay=self.hparams.weight_decay,
            )
        else:
            raise NotImplementedError(f"Optimizer {self.hparams.optimizer} is not implemented")

        return [optimizer_G, optimizer_D]
