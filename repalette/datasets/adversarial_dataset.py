from PIL import Image

from repalette.datasets import AbstractQueryDataset, PreTrainDataset
from repalette.utils.transforms import (
    sort_palette as sort_palette_by_hue,
)
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


class AdversarialRecolorDataset(PreTrainDataset):
    def _getitem(self, index):
        """
        @return: source_pair and target_pair - for generator; original_pair - for discriminator
        """
        (
            source_pair,
            target_pair,
        ) = super().__getitem__(index)

        randomizer = self.get_randomizer()

        random_idx = randomizer.randint(0, len(self.query) - 1)
        rgb_image = self.query[random_idx]

        original_image = Image.open(rgb_image.path)
        original_palette = rgb_image.palette

        if self.sort_palette:
            original_palette = sort_palette_by_hue(original_palette)

        original_palette = Image.fromarray(original_palette)

        [original_image_aug] = self.image_transform(original_image, 0)
        [original_palette_aug] = self.palette_transform(original_palette, 0)

        original_pair = (
            original_image_aug,
            original_palette_aug,
        )

        return (
            source_pair,
            target_pair,
            original_pair,
        )


class AdversarialDataset(AbstractQueryDataset):
    def __init__(
        self,
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
            random_seed=random_seed,
            train_kwargs=train_kwargs,
            test_kwargs=test_kwargs,
        )
        if sort_palette and shuffle_palette:
            raise ValueError("Don't sort and shuffle the palette at the same time!")

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

    def _getitem(self, index):
        """
        :param index: index of item to get from the dataset
        :return: image of shape [3, self.resize] and palette of shape [3, 1, 6]
        """
        randomizer = self.get_randomizer(random_seed=index)
        random_idx = randomizer.randint(0, len(self.query) - 1)

        original_rgb_image = self.query[index]
        random_rgb_image = self.query[random_idx]

        source_image = Image.open(original_rgb_image.path)
        original_image = Image.open(original_rgb_image.path)
        source_palette = original_rgb_image.palette
        target_palette = random_rgb_image.palette

        if self.sort_palette:
            target_palette = sort_palette_by_hue(target_palette)
            source_palette = sort_palette_by_hue(source_palette)

        target_palette = Image.fromarray(target_palette)
        source_palette = Image.fromarray(source_palette)

        (source_image_aug,) = self.image_transform(source_image, 0)
        (original_image_aug,) = self.image_transform(original_image, 0)

        (target_palette_aug,) = self.palette_transform(target_palette, 0)
        (source_palette,) = self.palette_transform(source_palette, 0)

        if self.shuffle_palette:
            target_palette_aug = target_palette_aug[:, :, torch.randperm(6)]
            source_palette = source_palette[:, :, torch.randperm(6)]

        return (source_image_aug, source_palette), (original_image_aug, target_palette_aug)
