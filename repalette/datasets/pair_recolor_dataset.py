from torch.utils.data import Dataset
from PIL import Image
import torch
from torchvision import transforms
import numpy as np
import random
from itertools import permutations
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from repalette.constants import IMAGE_SIZE, RGB_DATABASE_PATH
from repalette.utils.transforms import (
    FullTransform,
    sort_palette as sort_palette_by_hue,
)
from repalette.db.rgb import RGBImage


class PairRecolorDataset(Dataset):
    def __init__(
        self,
        multiplier: int = 16,
        query: list = None,
        shuffle=True,
        shuffle_palette=False,
        sort_palette=True,
        transform=None,
        normalize=True,
    ):
        """
        Dataset constructor.
        :param multiplier: an odd multiplier for color augmentation
        :param shuffle: if to shuffle images and color augmentation
        :param shuffle_palette: if to shuffle output palettes
        :param sort_palette: if to sort output palettes by hue
        :param transform: optional transform to be applied on a sample
        :param normalize: if to normalize LAB images to be in [-1, 1] range
        """
        if sort_palette and shuffle_palette:
            raise ValueError("Don't sort and shuffle the palette at the same time!")

        self.multiplier = multiplier
        self.shuffle_palette = shuffle_palette
        self.sort_palette = sort_palette
        self.normalize = normalize
        self.transform = transform

        hue_variants = np.linspace(-0.5, 0.5, self.multiplier)

        self.hue_pairs = [perm for perm in permutations(hue_variants, 2)]

        if transform is None:
            transform = transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    transforms.Resize(IMAGE_SIZE),
                ]
            )

        self.img_transform = FullTransform(transform, normalize=normalize)
        self.palette_transform = FullTransform(normalize=normalize)

        self.n_pairs = len(self.hue_pairs)

        if query is None:
            engine = create_engine(f"sqlite:///{RGB_DATABASE_PATH}")
            # create a configured "Session" class
            Session = sessionmaker(bind=engine)
            session = Session()
            self.query = session.query(RGBImage).all()
            session.close()
        else:
            self.query = query

        self.correct_order_query = self.query

        if shuffle:
            random.shuffle(self.hue_pairs)

    def __getitem__(self, index):
        """
        :param index: index of item to get from the dataset
        :return: image of shape [3, self.resize] and palette of shape [3, 1, 6]
        """
        pair_index = index % self.n_pairs
        hue_shift_first, hue_shift_second = self.hue_pairs[pair_index]
        i = (
            index // self.n_pairs
        )  # actual image index (from design-seeds-data directory)

        rgb_image = self.query[i]

        image = Image.open(rgb_image.path)
        palette = rgb_image.palette

        if self.sort_palette:
            palette = sort_palette_by_hue(palette)

        palette = Image.fromarray(palette)

        img_aug_first, img_aug_second = self.img_transform(
            image, hue_shift_first, hue_shift_second
        )
        palette_aug_first, palette_aug_second = self.palette_transform(
            palette, hue_shift_first, hue_shift_second
        )

        if self.shuffle_palette:
            palette_aug_first = palette_aug_first[:, :, torch.randperm(6)]
            palette_aug_second = palette_aug_second[:, :, torch.randperm(6)]

        return (img_aug_first, palette_aug_first), (img_aug_second, palette_aug_second)

    def __len__(self):
        """
        :return:
        """
        return len(self.query) * self.n_pairs

    def split(self, test_size=0.2, shuffle=True):
        query = self.query

        if shuffle:
            random.shuffle(query)

        train_query = query[: int(len(query) * (1 - test_size))]
        test_query = query[int(len(query) * (1 - test_size)) :]

        train = self.__class__(
            multiplier=self.multiplier,
            query=train_query,
            shuffle=shuffle,
            sort_palette=self.sort_palette,
            normalize=self.normalize,
            transform=self.transform,
            shuffle_palette=self.shuffle_palette,
        )
        test = self.__class__(
            multiplier=self.multiplier,
            query=test_query,
            shuffle=shuffle,
            sort_palette=self.sort_palette,
            normalize=self.normalize,
            transform=self.transform,
            shuffle_palette=self.shuffle_palette,
        )

        return train, test

    def shuffle(self, to_shuffle=True):
        """
        Shuffles data.
        :param to_shuffle: if to shuffle
        """
        if to_shuffle:
            random.shuffle(self.query)
            random.shuffle(self.hue_pairs)
        else:
            self.query = self.correct_order_query

        return self
