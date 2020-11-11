from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as TF
import torch
from torchvision.transforms import Resize
import numpy as np
import random
from itertools import permutations
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.sql.expression import func
from sqlalchemy.orm.query import Query

from repalette.constants import ROOT_DIR, IMAGE_SIZE, DATABASE_PATH
from repalette.utils.color import smart_hue_adjust
from repalette.utils.models import RawImage, RGBImage


class PairRecolorDataset(Dataset):
    def __init__(
        self,
        multiplier: int,
        query: Query = None,
        resize: tuple = IMAGE_SIZE,
        shuffle_palette=True,
    ):
        """
        Dataset constructor.
        :param multiplier: an odd multiplier for color augmentation
        :param resize: size to which the image will be resized with `torhvision.trainsforms.Resize`
        :param shuffle_palette: if to shuffle output palettes
        """
        # if multiplier % 2 == 0:
        #     raise ValueError("Multiplier must be odd.")
        self.multiplier = multiplier
        self.resize = resize
        self.shuffle_palette = shuffle_palette

        hue_variants = np.linspace(-0.5, 0.5, self.multiplier)
        self.hue_pairs = [perm for perm in permutations(hue_variants, 2)]
        self.n_pairs = len(self.hue_pairs)

        if query is None:
            engine = create_engine(f"sqlite:///{DATABASE_PATH}")
            # create a configured "Session" class
            Session = sessionmaker(bind=engine)
            session = Session()
            self.query = session.query(RGBImage)
            session.close()
        else:
            self.query = query

        self.query_mapping = list(range(self.query.count()))  # use this for shuffling

    def __getitem__(self, index):
        """
        :param index: index of item to get from the dataset
        :return: image of shape [3, self.resize] and palette of shape [3, 1, 6]
        """
        pair_index = index % self.n_pairs
        hue_shift_1, hue_shift_2 = self.hue_pairs[pair_index]
        i = (
            index // self.n_pairs
        )  # actual image index (from design-seeds-data directory)

        # user query mapping
        i = self.query_mapping[i]

        rgb_image = self.query[i]

        image = Image.open(rgb_image.path)

        if self.resize:
            resize = Resize(self.resize)
            image = resize(image)

        image_aug_1 = TF.to_tensor(smart_hue_adjust(image, hue_shift_1)).to(torch.float)
        image_aug_2 = TF.to_tensor(smart_hue_adjust(image, hue_shift_2)).to(torch.float)

        palette = Image.fromarray(rgb_image.palette)

        palette_aug_1 = TF.to_tensor(smart_hue_adjust(palette, hue_shift_1)).to(
            torch.float
        )
        palette_aug_2 = TF.to_tensor(smart_hue_adjust(palette, hue_shift_2)).to(
            torch.float
        )

        if self.shuffle_palette:
            palette_aug_1 = palette_aug_1[:, :, torch.randperm(6)]
            palette_aug_2 = palette_aug_2[:, :, torch.randperm(6)]

        return (image_aug_1, palette_aug_1), (image_aug_2, palette_aug_2)

    def __len__(self):
        """
        :return:
        """
        return self.query.count() * self.n_pairs

    def split(self, test_size=0.2, shuffle=True):
        all_ids = [item.id for item in self.query]

        if shuffle:
            random.shuffle(all_ids)

        train_ids = all_ids[:int(len(all_ids) * (1-test_size))]
        test_ids = all_ids[int(len(all_ids) * (1-test_size)):]

        train_query = self.query.filter(RGBImage.id.in_(train_ids))
        test_query = self.query.filter(RGBImage.id.in_(test_ids))

        train = PairRecolorDataset(multiplier=self.multiplier, query=train_query)
        test = PairRecolorDataset(multiplier=self.multiplier, query=test_query)

        train.shuffle(shuffle)
        test.shuffle(shuffle)

        return train, test

    def shuffle(self, to_shuffle=True):
        """
        Shuffles data.
        :param to_shuffle: if to shuffle
        """
        if to_shuffle:
            random.shuffle(self.query_mapping)
        else:
            self.query_mapping = list(range(self.query.count()))

        return self
