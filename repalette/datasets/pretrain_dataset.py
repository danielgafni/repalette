from torch.utils.data import Dataset
from PIL import Image
import torch
from torchvision import transforms
import numpy as np
from random import Random
from itertools import permutations
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from repalette.constants import (
    DEFAULT_IMAGE_SIZE,
    RGB_DATABASE_PATH,
)
from repalette.utils.transforms import (
    FullTransform,
    sort_palette as sort_palette_by_hue,
)
from repalette.db.rgb import RGBImage
from repalette.datasets import AbstractRecolorDataset


class PreTrainDataset(AbstractRecolorDataset):
    def __init__(
        self,
        multiplier: int = 16,
        query: list = None,
        shuffle=True,
        shuffle_palette=False,
        sort_palette=True,
        transform=None,
        normalize=True,
        image_size=DEFAULT_IMAGE_SIZE,
        random_seed=None,
        train_kwargs=None,
        test_kwargs=None,
    ):
        """
        :param multiplier: an odd multiplier for color augmentation
        :param shuffle: if to shuffle images and color augmentation
        :param shuffle_palette: if to shuffle output palettes
        :param sort_palette: if to sort output palettes by hue
        :param transform: optional transform to be applied on a sample
        :param normalize: if to normalize LAB images to be in [-1, 1] range
        :param image_size: image size to resize to (lol)
        :param random_seed: random seed for shuffling and splitting
        """
        if transform is None:
            transform = transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    transforms.Resize(image_size),
                ]
            )

        if train_kwargs is None:
            train_kwargs = dict(shuffle=True, transform=transform)
        if test_kwargs is None:
            test_kwargs = dict(shuffle=False, transform=transforms.Resize(image_size))

        if query is None:
            engine = create_engine(f"sqlite:///{RGB_DATABASE_PATH}")
            # create a configured "Session" class
            Session = sessionmaker(bind=engine)
            session = Session()
            query = session.query(RGBImage).all()
            session.close()

        super().__init__(
            query=query,
            multiplier=multiplier,
            random_seed=random_seed,
            train_kwargs=train_kwargs,
            test_kwargs=test_kwargs,
        )
        if sort_palette and shuffle_palette:
            raise ValueError("Don't sort and shuffle the palette at the same time!")

        self.multiplier = multiplier
        self.shuffle_palette = shuffle_palette
        self.sort_palette = sort_palette
        self.normalize = normalize
        self.transform = transform
        self.image_size = image_size

        self.image_transform = FullTransform(transform, normalize=normalize)
        self.palette_transform = FullTransform(normalize=normalize)

        if shuffle:
            randomizer = self.get_randomizer(random_seed=random_seed)
            randomizer.shuffle(self.query)
            randomizer.shuffle(self.hue_pairs)

    def _make_hue_pairs(self, multiplier):
        hue_variants = np.linspace(-0.5, 0.5, self.multiplier)
        hue_pairs = [perm for perm in permutations(hue_variants, 2)]

        return hue_pairs

    def _getitem(self, index):
        """
        :param index: index of item to get from the dataset
        :return: image of shape [3, self.resize] and palette of shape [3, 1, 6]
        """
        pair_index = index % self.n_pairs
        (
            hue_shift_first,
            hue_shift_second,
        ) = self.hue_pairs[pair_index]
        i = index // self.n_pairs  # actual image index (from design-seeds-data directory)

        rgb_image = self.query[i]

        image = Image.open(rgb_image.path)
        palette = rgb_image.palette

        if self.sort_palette:
            palette = sort_palette_by_hue(palette)

        palette = Image.fromarray(palette)

        (img_aug_first, img_aug_second,) = self.image_transform(
            image,
            hue_shift_first,
            hue_shift_second,
        )
        (palette_aug_first, palette_aug_second,) = self.palette_transform(
            palette,
            hue_shift_first,
            hue_shift_second,
        )

        if self.shuffle_palette:
            palette_aug_first = palette_aug_first[:, :, torch.randperm(6)]
            palette_aug_second = palette_aug_second[:, :, torch.randperm(6)]

        return (
            img_aug_first,
            palette_aug_first,
        ), (img_aug_second, palette_aug_second)
