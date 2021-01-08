from pytorch_lightning.callbacks import Callback
import torch.nn as nn
import torch
import abc

from repalette.utils.visualization import lab_batch_to_rgb_image_grid
from repalette.lightning.systems import PreTrainSystem


class LogRecoloringToTensorboard(Callback):
    """
    Logs a batch of images, target images, target palettes and recolored images with TensorBoardLogger
    """

    def __init__(self, batches=6):
        self.batches = batches

    def on_train_epoch_end(self, trainer, pl_module, outputs):
        train_dataloader = pl_module.train_dataloader()

        self._log_recoloring(pl_module, train_dataloader, "Train", True)
        # self._log_recoloring(pl_module, train_dataloader, "Train", False)  # this doesn't work for some reason

    def on_validation_epoch_end(self, trainer, pl_module):
        val_dataloader = pl_module.val_dataloader()

        self._log_recoloring(pl_module, val_dataloader, "Val", False)
        # self._log_recoloring(pl_module, val_dataloader, "Val", False)  # this doesn't work for some reason

    def on_test_epoch_end(self, trainer, pl_module):
        test_dataloader = pl_module.test_dataloader()

        self._log_recoloring(pl_module, test_dataloader, "Test", False)

    def _log_recoloring(self, pl_module: PreTrainSystem, dataloader, stage, to_shuffle):
        if to_shuffle:
            prefix = "random_"
        else:
            prefix = "persistent_"
        pl_module.scaler.to(pl_module.device)

        original_images = []
        target_images = []
        recolored_images = []
        target_palettes = []

        iter_dataloader = iter(dataloader)

        batch_size = 0

        for _ in range(self.batches):
            original_img, target_img, target_palette = self.get_data_from_iter_dataloader(iter_dataloader)
            original_img = original_img.to(pl_module.device)
            target_img = target_img.to(pl_module.device)
            target_palette = target_palette.to(pl_module.device)

            with torch.no_grad():
                _target_palette = nn.Flatten()(target_palette)
                recolored_img = pl_module.generator(original_img, _target_palette)

            original_luminance = original_img.clone()[:, 0:1, ...].to(pl_module.device)
            recolored_img_with_luminance = torch.cat(
                (original_luminance, recolored_img), dim=1
            )

            original_img = pl_module.scaler.inverse_transform(original_img)
            target_img = pl_module.scaler.inverse_transform(target_img)
            target_palette = pl_module.scaler.inverse_transform(target_palette)
            recolored_img_with_luminance = pl_module.scaler.inverse_transform(
                recolored_img_with_luminance
            )

            original_images.append(original_img)
            target_images.append(target_img)
            recolored_images.append(recolored_img_with_luminance)
            target_palettes.append(target_palette)

            batch_size = original_img.size(0)

        original_images = torch.cat(original_images, dim=0)
        target_images = torch.cat(target_images, dim=0)
        recolored_images = torch.cat(recolored_images, dim=0)
        target_palettes = torch.cat(target_palettes, dim=0)

        original_grid = lab_batch_to_rgb_image_grid(original_images, nrow=batch_size)
        target_grid = lab_batch_to_rgb_image_grid(target_images, nrow=batch_size)

        target_palette_img = target_palettes.view(-1, 3, 6, 1)
        target_palette_grid = lab_batch_to_rgb_image_grid(
            target_palette_img, nrow=batch_size, pad_value=1.0, padding=1
        )

        recolored_grid = lab_batch_to_rgb_image_grid(recolored_images, nrow=batch_size)

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

    @abc.abstractmethod
    def get_data_from_iter_dataloader(self, iter_dataloader):
        pass


class LogPairRecoloringToTensorboard(LogRecoloringToTensorboard):
    """
    Log recolored images for PairRecolorDataset
    """
    def get_data_from_iter_dataloader(self, iter_dataloader):
        (original_img, original_palette), (target_img, target_palette) = next(iter_dataloader)
        return original_img, target_img, target_palette


class LogTripletRecoloringToTensorboard(LogRecoloringToTensorboard):
    def get_data_from_iter_dataloader(self, iter_dataloader):
        (
            (source_img, source_palette),
            (target_img, target_palette),
            (original_img, original_palette),
        ) = next(iter_dataloader)

        return source_img, target_img, target_palette
