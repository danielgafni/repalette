import argparse
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import wandb
from dotenv import load_dotenv

from repalette.models.palette_net import PaletteNet
from repalette.utils.visualization import lab_batch_to_rgb_image_grid
from repalette.utils.transforms import Scaler

from repalette.constants import (
    DEFAULT_PRETRAIN_LR,
    DEFAULT_PRETRAIN_BETA_1,
    DEFAULT_PRETRAIN_BETA_2,
    S3_LIGHTNING_LOGS_DIR,
    S3_MODEL_CHECKPOINTS_DIR,
    DEFAULT_PRETRAIN_WEIGHT_DECAY,
    LIGHTNING_LOGS_DIR
)
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.metrics.regression import MeanSquaredError
from repalette.lightning.datamodules import PreTrainDataModule


class PreTrainTask(pl.LightningModule):
    """
    Wrapper for pre-training of PaletteNet.
    """

    def __init__(self, learning_rate, beta_1, beta_2, weight_decay, optimizer, batch_size, multiplier):
        super().__init__()

        self.save_hyperparameters()

        self.model = PaletteNet()
        self.MSE = MeanSquaredError()
        self.scaler = Scaler()

        self.datamodule = None
        self.loggerer_provider = None

    def set_datamodule(self, datamodule):
        self.datamodule = datamodule

    def set_logger_provider(self, logger_provider):
        if logger_provider not in ["tensorboard", "wandb"]:
            raise NotImplementedError(f"Logger provider {logger_provider} is not implemented")
        self.loggerer_provider = logger_provider

    def forward(self, img, palette):
        return self.model(img, palette)

    def training_step(self, batch, batch_idx):
        (source_img, _), (target_img, target_palette) = batch
        target_palette = nn.Flatten()(target_palette)
        recolored_img_ab = self.model(source_img, target_palette)
        mse_loss = self.MSE(recolored_img_ab, target_img[:, 1:, :, :])
        self.log("Train/loss_step", mse_loss)
        return mse_loss

    def validation_step(self, batch, batch_idx):
        (source_img, _), (target_img, target_palette) = batch
        target_palette = nn.Flatten()(target_palette)
        recolored_img_ab = self.model(source_img, target_palette)
        loss = self.MSE(recolored_img_ab, target_img[:, 1:, :, :])
        return loss

    def training_epoch_end(self, outputs):
        # log training loss
        self.log("Train/loss_epoch", self.MSE.compute())

        # visualize images
        (original_img, _), (target_img, target_palette) = next(
            iter(self.train_dataloader())
        )

        original_img = original_img.to(self.device)
        target_img = target_img.to(self.device)
        target_palette = target_palette.to(self.device)

        with torch.no_grad():
            _target_palette = nn.Flatten()(target_palette)
            recolored_img = self.model(original_img, _target_palette)

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

        self.log_recoloring(original_grid, target_grid, target_palette_grid, recolored_grid)

    def validation_epoch_end(self, outputs):
        # log validation loss
        self.log("Val/loss_epoch", self.MSE.compute())

        # visualize images
        self.val_dataloader().shuffle(True)
        (original_img, _), (target_img, target_palette) = next(
            iter(self.val_dataloader())
        )
        self.val_dataloader().shuffle(False)

        original_img = original_img.to(self.device)
        target_img = target_img.to(self.device)
        target_palette = target_palette.to(self.device)

        with torch.no_grad():
            _target_palette = nn.Flatten()(target_palette)
            recolored_img = self.model(original_img, _target_palette)

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

        self.log_recoloring(original_grid, target_grid, target_palette_grid, recolored_grid)

        # self.logger.log_hyperparams(self.hparams)

    def train_dataloader(self):
        return self.datamodule.train_dataloader()

    def val_dataloader(self):
        return self.datamodule.val_dataloader()

    def test_dataloader(self):
        return self.datamodule.test_dataloader()

    def configure_optimizers(self):
        if self.hparams.optimizer == "adam":
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.hparams.learning_rate,
                betas=(self.hparams.beta_1, self.hparams.beta_2),
            )
        elif self.hparams.optimizer == "adamw":
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.hparams.learning_rate,
                betas=(self.hparams.beta_1, self.hparams.beta_2),
            )
        else:
            raise NotImplementedError(
                f"Optimizer {self.hparams.optimizer} is not implemented"
            )
        return optimizer

    def log_recoloring(self, original_grid, target_grid, target_palette_grid, recolored_grid):
        if self.loggerer_provider == "tensorboard":
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
        elif self.loggerer_provider == "wandb":
            self.logger.experiment.log({"train_images": [
                wandb.Image(original_grid, caption="Train/Original"),
                wandb.Image(target_grid, caption="Train/Target"),
                wandb.Image(target_palette_grid, caption="Train/Target_Palette"),
                wandb.Image(recolored_grid, caption="Train/Recolored")
            ]})


if __name__ == "__main__":
    # load .env variables
    load_dotenv()

    # hyperparameters
    hparams_parser = argparse.ArgumentParser()

    # trainer
    hparams_parser.add_argument("--gpus", type=int, default=1)
    hparams_parser.add_argument(
        "--precision", type=int, default=16, choices=[16, 32]
    )
    hparams_parser.add_argument(
        "--accumulate-grad-batches", type=int, default=1
    )
    hparams_parser.add_argument(
        "--gradient-clip-val", type=float, default=0.
    )
    hparams_parser.add_argument(
        "--auto-lr-find", type=str, default=False, choices=[False, "learning_rate"]
    )
    hparams_parser.add_argument(
            "--auto-scale-batch-size", type=str, default=None, choices=[None, "power", "binsearch"]
        )

    # callbacks
    hparams_parser.add_argument("--patience", type=int, default=20)
    hparams_parser.add_argument("--save-top-k", type=int, default=0)

    # pretrain task
    hparams_parser.add_argument(
        "--learning-rate", type=float, default=DEFAULT_PRETRAIN_LR
    )
    hparams_parser.add_argument(
        "--beta-1", type=float, default=DEFAULT_PRETRAIN_BETA_1
    )
    hparams_parser.add_argument(
        "--beta-2", type=float, default=DEFAULT_PRETRAIN_BETA_2
    )
    hparams_parser.add_argument(
        "--weight-decay", type=float, default=DEFAULT_PRETRAIN_WEIGHT_DECAY
    )
    hparams_parser.add_argument(
        "--optimizer", type=str, default="adam", choices=["adam", "adamw"]
    )
    hparams_parser.add_argument("--batch-size", type=int, default=8)
    hparams_parser.add_argument("--multiplier", type=int, default=16)

    # datamodule
    hparams_parser.add_argument("--num-workers", type=int, default=7)
    hparams_parser.add_argument("--shuffle", type=bool, default=True)
    hparams_parser.add_argument("--size", type=float, default=1.)
    hparams_parser.add_argument("--pin-memory", type=bool, default=True)

    # misc
    hparams_parser.add_argument(
        "--name", type=str, default="test", help="experiment name"
    )
    hparams_parser.add_argument(
        "--logger", type=str, default="tensorboard", choices=["tensorboard", "wandb"]
    )

    hparams = hparams_parser.parse_args()

    # main LightningModule
    pretrain_task = PreTrainTask(learning_rate=hparams.learning_rate,
                                 beta_1=hparams.beta_1,
                                 beta_2=hparams.beta_2,
                                 weight_decay=hparams.weight_decay,
                                 optimizer=hparams.optimizer,
                                 batch_size=hparams.batch_size,
                                 multiplier=hparams.multiplier)

    pretrain_checkpoints = ModelCheckpoint(
        dirpath=S3_MODEL_CHECKPOINTS_DIR,
        monitor="Val/Loss",
        verbose=True,
        mode="min",
        save_top_k=hparams.save_top_k,
    )

    pretrain_early_stopping = EarlyStopping(
        monitor="Val/Loss",
        min_delta=0.00,
        patience=hparams.patience,
        verbose=False,
        mode="min",
    )

    logger = None
    if hparams.logger == "tensorboard":
        logger = TensorBoardLogger(S3_LIGHTNING_LOGS_DIR, name=hparams.name)
    elif hparams.logger == "wandb":
        logger = WandbLogger(save_dir=LIGHTNING_LOGS_DIR, project="repalette", name=hparams.name)

    trainer = Trainer.from_argparse_args(
        hparams,
        logger=logger,
        checkpoint_callback=pretrain_checkpoints,
        callbacks=[pretrain_early_stopping],
        auto_lr_find=hparams.auto_lr_find,
    )

    datamodule = PreTrainDataModule(
        batch_size=pretrain_task.hparams.batch_size,
        multiplier=pretrain_task.hparams.multiplier,
        shuffle=hparams.shuffle,
        num_workers=hparams.num_workers,
        size=hparams.size,
        pin_memory=hparams.pin_memory
    )

    pretrain_task.set_datamodule(datamodule)
    pretrain_task.set_logger_provider(hparams.logger)

    trainer.tune(pretrain_task)

    trainer.fit(pretrain_task)
