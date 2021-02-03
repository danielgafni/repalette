from argparse import ArgumentParser

import pytorch_lightning as pl
from torchvision import transforms

from repalette.constants import DEFAULT_IMAGE_SIZE, DEFAULT_PRETRAIN_BATCH_SIZE
from repalette.datasets import GANDataset, PreTrainDataset
from repalette.datasets.utils import ShuffleDataLoader


class PreTrainDataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size=DEFAULT_PRETRAIN_BATCH_SIZE,
        multiplier=16,
        shuffle=True,
        num_workers=15,
        transform=None,
        image_size=DEFAULT_IMAGE_SIZE,
        size=1,
        pin_memory=True,
        train_batch_from_same_image=False,
        val_batch_from_same_image=False,
        test_batch_from_same_image=False,
        **kwargs,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.multiplier = multiplier
        self.num_workers = num_workers
        self.size = size
        self.pin_memory = pin_memory
        self.shuffle = shuffle
        self.train_batch_from_same_image = train_batch_from_same_image
        self.val_batch_from_same_image = val_batch_from_same_image
        self.test_batch_from_same_image = test_batch_from_same_image

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
        data = PreTrainDataset(
            multiplier=self.multiplier,
            shuffle=self.shuffle,
            transform=self.transform,
        )
        data, _ = data.split(
            test_size=(1 - self.size),
            shuffle=True,
        )
        train, val = data.split(test_size=0.2, shuffle=True)
        val, test = val.split(test_size=0.5, shuffle=True)

        self.train = train
        self.val = val
        self.test = test

    def train_dataloader(self):
        train_dataloader = ShuffleDataLoader(
            self.train,
            shuffle=not self.train_batch_from_same_image,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            pin_memory=self.pin_memory,
        )
        # train dataloader should be shuffled!
        train_dataloader.shuffle(True)  # this will make no difference if self.train_batch_from_same_image == True
        return train_dataloader

    def val_dataloader(self):
        val_dataloader = ShuffleDataLoader(
            self.val,
            shuffle=not self.val_batch_from_same_image,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            pin_memory=self.pin_memory,
        )
        return val_dataloader

    def test_dataloader(self):
        test_dataloader = ShuffleDataLoader(
            self.test,
            shuffle=not self.test_batch_from_same_image,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            pin_memory=self.pin_memory,
        )
        return test_dataloader

    @staticmethod
    def add_argparse_args(parent_parser: ArgumentParser) -> ArgumentParser:
        hparams_parser = ArgumentParser(parents=[parent_parser], add_help=False)

        hparams_parser.add_argument("--batch-size", type=int, default=8)
        hparams_parser.add_argument("--multiplier", type=int, default=16)
        hparams_parser.add_argument("--num-workers", type=int, default=7)
        hparams_parser.add_argument("--shuffle", type=bool, default=True)
        hparams_parser.add_argument("--size", type=float, default=1.0)
        hparams_parser.add_argument("--pin-memory", type=bool, default=True)
        hparams_parser.add_argument("--train-batch-from-same-image", type=bool, default=False)
        hparams_parser.add_argument("--val-batch-from-same-image", type=bool, default=True)
        hparams_parser.add_argument("--test-batch-from-same-image", type=bool, default=True)

        return hparams_parser

    # don't uncomment!!!
    # def transfer_batch_to_device(self, batch: Any, device: torch.device) -> Any:
    #     # maybe we want this later
    #
    # def prepare_data(self, *args, **kwargs):
    #     # maybe we want this later


class GANDataModule(PreTrainDataModule):
    def setup(self, stage=None):
        data = GANDataset(
            multiplier=self.multiplier,
            shuffle=self.shuffle,
            transform=self.transform,
        )
        data, _ = data.split(
            test_size=(1 - self.size),
            shuffle=True,
        )
        train, val = data.split(test_size=0.2, shuffle=True)
        val, test = val.split(test_size=0.5, shuffle=True)

        self.train = train
        self.val = val
        self.test = test
