from typing import Any

import pytorch_lightning as pl
import torch
from torchvision import transforms
from repalette.datasets import PairRecolorDataset
from repalette.datasets.utils import ShuffleDataLoader
from repalette.constants import IMAGE_SIZE


class PreTrainDataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size=8,
        multiplier=16,
        shuffle=True,
        num_workers=8,
        transform=None,
        image_size=IMAGE_SIZE,
        size=1,
        pin_memory=True,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.multiplier = multiplier
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.size = size
        self.pin_memory = pin_memory

        self.train = None
        self.val = None
        self.test = None

        # transform
        if transform is None:
            transform = transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    transforms.Resize(image_size),
                ]
            )
        self.transform = transform

    def setup(self, stage=None):
        data = PairRecolorDataset(
            multiplier=self.multiplier, shuffle=self.shuffle, transform=self.transform
        )
        data, _ = data.split(test_size=(1 - self.size), shuffle=True)
        train, val = data.split(test_size=0.2, shuffle=True)
        val, test = val.split(test_size=0.5, shuffle=True)

        self.train = train
        self.val = val
        self.test = test

    def train_dataloader(self):
        train_dataloader = ShuffleDataLoader(
            self.train,
            shuffle=False,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            pin_memory=self.pin_memory,
        )
        # train dataloader should be shuffled!
        train_dataloader.shuffle(True)
        return train_dataloader

    def val_dataloader(self):
        val_dataloader = ShuffleDataLoader(
            self.val,
            shuffle=False,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            pin_memory=self.pin_memory,
        )
        return val_dataloader

    def test_dataloader(self):
        test_dataloader = ShuffleDataLoader(
            self.test,
            shuffle=False,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            pin_memory=self.pin_memory,
        )
        return test_dataloader

    # don't uncomment!!!
    # def transfer_batch_to_device(self, batch: Any, device: torch.device) -> Any:
    #     # maybe we want this later
    #
    # def prepare_data(self, *args, **kwargs):
    #     # maybe we want this later
