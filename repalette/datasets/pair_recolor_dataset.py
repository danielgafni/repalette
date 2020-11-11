from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as TF
import torch
from torchvision.transforms import Resize
import numpy as np
from pandas import DataFrame
from itertools import permutations
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from repalette.constants import ROOT_DIR, IMAGE_SIZE, DATABASE_PATH
from repalette.utils.color import smart_hue_adjust
from repalette.utils.models import RawImage, RGBImage


class PairRecolorDataset(Dataset):
    def __init__(
        self,
        multiplier: int,
        path_prefix: str = ROOT_DIR,
        resize: tuple = IMAGE_SIZE,
        shuffle_palette=True,
    ):
        """
        Dataset constructor.
        :param multiplier: an odd multiplier for color augmentation
        :param path_prefix: full path prefix to add before relative paths in data
        :param resize: size to which the image will be resized with `torhvision.trainsforms.Resize`
        :param shuffle_palette: if to shuffle output palettes
        """
        # if multiplier % 2 == 0:
        #     raise ValueError("Multiplier must be odd.")
        self.multiplier = multiplier
        self.path_prefix = path_prefix
        self.resize = resize
        self.shuffle_palette = shuffle_palette

        hue_variants = np.linspace(-0.5, 0.5, self.multiplier)
        self.hue_pairs = [perm for perm in permutations(hue_variants, 2)]
        self.n_pairs = len(self.hue_pairs)

        engine = create_engine(f"sqlite:///{DATABASE_PATH}")
        # create a configured "Session" class
        Session = sessionmaker(bind=engine)
        session = Session()

        self.query = session.query(RGBImage)

        session.close()

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

    def shuffle(self, set_shuffle=True):
        """
        Shuffles data.
        :param set_shuffle: set data to shuffled or unshuffled state
        """
        if set_shuffle:
            self.data = self.data.sample(frac=1)

        else:
            self.data = self.data.reindex(list(range(len(self.data))))

        return self
