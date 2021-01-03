from pytorch_lightning.callbacks import Callback
import torch.nn as nn
import torch

from repalette.constants import DISCORD_TRAINING_CHANNEL_ID
from repalette.utils.visualization import lab_batch_to_rgb_image_grid
from repalette.lightning.systems import PreTrainSystem
from repalette.utils.notify import notify_discord


class LogRecoloringToTensorboard(Callback):
    """
    Logs a batch of images, target images, target palettes and recoloder images with TensorBoardLogger
    """

    def on_train_epoch_end(self, trainer, pl_module, outputs):
        train_dataloader = pl_module.train_dataloader()

        self._log_recoloring(pl_module, train_dataloader, "Train", True)
        # self._log_recoloring(pl_module, train_dataloader, "Train", False)  # this doesn't work for some reason

    def on_validation_epoch_end(self, trainer, pl_module):
        val_dataloader = pl_module.val_dataloader()

        self._log_recoloring(pl_module, val_dataloader, "Val", True)
        # self._log_recoloring(pl_module, val_dataloader, "Val", False)  # this doesn't work for some reason

    def on_test_epoch_end(self, trainer, pl_module):
        test_dataloader = pl_module.val_dataloader()

        self._log_recoloring(pl_module, test_dataloader, "Test", True)

    @staticmethod
    def _log_recoloring(pl_module: PreTrainSystem, dataloader, stage, to_shuffle):
        if to_shuffle:
            prefix = "random_"
        else:
            prefix = "persistent_"
        dataloader.shuffle(to_shuffle)
        (original_img, _), (target_img, target_palette) = next(iter(dataloader))

        original_img = original_img.to(pl_module.device)
        target_img = target_img.to(pl_module.device)
        target_palette = target_palette.to(pl_module.device)

        with torch.no_grad():
            _target_palette = nn.Flatten()(target_palette)
            recolored_img = pl_module.generator(original_img, _target_palette)

        original_luminance = original_img.clone()[:, 0:1, ...].to(pl_module.device)
        recolored_img_with_luminance = torch.cat((original_luminance, recolored_img), dim=1)

        pl_module.scaler.to(pl_module.device)

        original_img = pl_module.scaler.inverse_transform(original_img)
        target_img = pl_module.scaler.inverse_transform(target_img)
        target_palette = pl_module.scaler.inverse_transform(target_palette)
        recolored_img_with_luminance = pl_module.scaler.inverse_transform(
            recolored_img_with_luminance
        )

        original_grid = lab_batch_to_rgb_image_grid(original_img)
        target_grid = lab_batch_to_rgb_image_grid(target_img)

        target_palette_img = target_palette.view(-1, 3, 6, 1)
        target_palette_grid = lab_batch_to_rgb_image_grid(
            target_palette_img, pad_value=1.0, padding=1
        )

        recolored_grid = lab_batch_to_rgb_image_grid(recolored_img_with_luminance)

        pl_module.logger.experiment.add_image(
            f"{stage}/{prefix}original", original_grid, pl_module.current_epoch
        )
        pl_module.logger.experiment.add_image(
            f"{stage}/{prefix}target", target_grid, pl_module.current_epoch
        )
        pl_module.logger.experiment.add_image(
            f"{stage}/{prefix}target_palette",
            target_palette_grid,
            pl_module.current_epoch,
        )
        pl_module.logger.experiment.add_image(
            f"{stage}/{prefix}recolored", recolored_grid, pl_module.current_epoch
        )


class LogHPMetric(Callback):
    def on_train_end(self, trainer, pl_module):
        pass


class NotifyTestEnd(Callback):
    def __init__(self, notifier="discord"):
        if notifier == "discord":
            self.do_notify = self._notify_discord
        else:
            raise NotImplementedError(f"notifier {notifier} is not implemented.")

    def on_test_end(self, trainer, pl_module):
        message = f"✨✨✨ Training of {trainer.version} has finished ✨✨✨"
        await self.do_notify(message=message)

    @staticmethod
    async def _notify_discord(message):
        await notify_discord(channel_id=DISCORD_TRAINING_CHANNEL_ID, message=message)
