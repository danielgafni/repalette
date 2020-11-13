from torch.utils.data import Dataset
from PIL import Image
import torch
from torchvision import transforms
import numpy as np
import random
from itertools import permutations
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.orm.query import Query

from repalette.constants import DATABASE_PATH
from repalette.utils.color import FullTransform
from repalette.utils.models import RGBImage


class PairRecolorDataset(Dataset):
    def __init__(
            self,
            multiplier: int = 16,
            query: Query = None,
            shuffle_palette=True,
            transform=None,
            normalize=True
    ):
        """
        Dataset constructor.
        :param multiplier: an odd multiplier for color augmentation
        :param shuffle_palette: if to shuffle output palettes
        :param transform: optional transform to be applied on a sample
        :param normalize: if to normalize LAB images to be in [-1, 1] range
        """
        self.multiplier = multiplier
        self.shuffle_palette = shuffle_palette
        self.normalize = normalize

        hue_variants = np.linspace(-0.5, 0.5, self.multiplier)

        self.hue_pairs = [perm for perm in permutations(hue_variants, 2)]
        if shuffle_palette:
            random.shuffle((self.hue_pairs))

        self.n_pairs = len(self.hue_pairs)

        self.transform = transform
        if transform is None:
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
            ])

        if query is None:
            engine = create_engine(f"sqlite:///{DATABASE_PATH}")
            # create a configured "Session" class
            Session = sessionmaker(bind=engine)
            session = Session()
            self.query = session.query(RGBImage).all()
            session.close()
        else:
            self.query = query

        self.correct_order_query = self.query

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

        img_transform = FullTransform(self.transform, normalize=self.normalize)
        palette_transform = FullTransform(normalize=self.normalize)

        image = Image.open(rgb_image.path)
        img_aug_first, img_aug_second = img_transform(image, hue_shift_first, hue_shift_second)

        palette = Image.fromarray(rgb_image.palette)
        palette_aug_first, palette_aug_second = palette_transform(palette, hue_shift_first,
                                                                  hue_shift_second)

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
            random.shuffle(self.hue_pairs)

        train_query = query[:int(len(query) * (1 - test_size))]
        test_query = query[int(len(query) * (1 - test_size)):]

        train = PairRecolorDataset(multiplier=self.multiplier, query=train_query)
        test = PairRecolorDataset(multiplier=self.multiplier, query=test_query)

        return train, test

    def shuffle(self, to_shuffle=True):
        """
        Shuffles data.
        :param to_shuffle: if to shuffle
        """
        if to_shuffle:
            random.shuffle(self.query)
        else:
            self.query = self.correct_order_query

        return self
