from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as TF
from torchvision.transforms import Resize
import numpy as np
from pandas import DataFrame
from repalette.constants import ROOT_DIR
from repalette.utils.color import smart_hue_adjust
from itertools import combinations


class RecolorDataset(Dataset):
    def __init__(self, data: DataFrame, multiplier: int, path_prefix: str = ROOT_DIR, resize=None):
        """
        Dataset constructor.
        :param data: DataFrame containing columns `image_path` and `palette_path`
        :param multiplier: an odd multiplier for color augmentation
        :param path_prefix: full path prefix to add before relative paths in data
        :param resize: size to which the image will be resized with `torhvision.trainsforms.Resize`
        """
        if multiplier % 2 == 0:
            raise ValueError("Multiplier must be odd.")
        self.multiplier = multiplier
        self.path_prefix = path_prefix
        self.resize = resize
        self.data = data

    def __getitem__(self, index):
        """
        :param index: index of item to get from the dataset
        :return: image of shape [3, self.resize] and palette of shape [3, 1, 6]
        """
        hue_shift = (index % self.multiplier - (self.multiplier - 1) / 2) / (self.multiplier - 1)
        i = index // self.multiplier  # actual image index (from design-seeds-data directory)

        image = Image.open(self.path_prefix + self.data["image_path"].iloc[i])
        if self.resize:
            resize = Resize(self.resize)
            image = resize(image)
        image_aug = TF.to_tensor(smart_hue_adjust(image, hue_shift))

        palette = Image.fromarray(np.load(self.path_prefix + self.data["palette_path"].iloc[i]).astype(np.uint8))
        palette_aug = TF.to_tensor(smart_hue_adjust(palette, hue_shift, lab=False))

        return image_aug, palette_aug

    def __len__(self):
        """
        :return:
        """
        return len(self.data) * self.multiplier


class PairRecolorDataset(Dataset):
    def __init__(self, data: DataFrame, multiplier: int, path_prefix: str = ROOT_DIR, resize=None):
        """
        Dataset constructor.
        :param data: DataFrame containing columns `image_path` and `palette_path`
        :param multiplier: an odd multiplier for color augmentation
        :param path_prefix: full path prefix to add before relative paths in data
        :param resize: size to which the image will be resized with `torhvision.trainsforms.Resize`
        """
        if multiplier % 2 == 0:
            raise ValueError("Multiplier must be odd.")
        self.multiplier = multiplier
        self.path_prefix = path_prefix
        self.resize = resize
        self.data = data

        hue_variants = np.linspace(-0.5, 0.5, self.multiplier)
        self.hue_pairs = [comb for comb in combinations(hue_variants, 2)]
        self.n_pairs = len(self.hue_pairs)

    def __getitem__(self, index):
        """
        :param index: index of item to get from the dataset
        :return: image of shape [3, self.resize] and palette of shape [3, 1, 6]
        """
        pair_index = index % self.n_pairs
        hue_shift_1, hue_shift_2 = self.hue_pairs[pair_index]
        i = index // self.n_pairs  # actual image index (from design-seeds-data directory)

        image = Image.open(self.path_prefix + self.data["image_path"].iloc[i])
        if self.resize:
            resize = Resize(self.resize)
            image = resize(image)
        image_aug_1 = TF.to_tensor(smart_hue_adjust(image, hue_shift_1))
        image_aug_2 = TF.to_tensor(smart_hue_adjust(image, hue_shift_2))

        palette = Image.fromarray(np.load(self.path_prefix + self.data["palette_path"].iloc[i]).astype(np.uint8))
        palette_aug_1 = TF.to_tensor(smart_hue_adjust(palette, hue_shift_1, lab=False))
        palette_aug_2 = TF.to_tensor(smart_hue_adjust(palette, hue_shift_2, lab=False))

        return (image_aug_1, palette_aug_1), (image_aug_2, palette_aug_2)

    def __len__(self):
        """
        :return:
        """
        return len(self.data) * self.n_pairs
