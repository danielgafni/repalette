import numpy as np
import torch
import torchvision.transforms.functional as TF

from torchvision.transforms import Resize
from torch.utils.data import Dataset
from PIL import Image
from pandas import DataFrame
from skimage.color import rgb2lab

from repalette.constants import ROOT_DIR
from repalette.utils.color import smart_hue_adjust


class RecolorDataset(Dataset):
    def __init__(self, data: DataFrame, multiplier: int, path_prefix: str = ROOT_DIR, resize=None):
        """
        Dataset constructor.
        :param data: DataFrame containing columns `image_path` and `palette_path`
        :param multiplier: an odd multiplier for color augmentation
        :param path_prefix: full path prefix to add before relative paths in data
        :param lab: if True, returns images in LAB format
        :param resize: size to which the image will be resized with `torhvision.trainsforms.Resize`
        """
        if multiplier % 2 == 0:
            raise ValueError("Multiplier must be odd.")
        self.multiplier = multiplier
        self.path_prefix = path_prefix
        self.resize = resize
        self.data = data
        self.lab = lab

    def __getitem__(self, index):
        """
        :param index: index of item to get from the dataset
        :return: image of shape [3, self.resize] and palette of shape [3, 1, 6]
        """
        hue_shift = (index % self.multiplier - (self.multiplier - 1) / 2) / (self.multiplier - 1)
        i = index // self.multiplier  # actual image index (from design-seeds-data directory)
        random_hue_shift = (np.random.randint(self.multiplier) % self.multiplier
                            - (self.multiplier - 1) / 2) / (self.multiplier - 1)  # pick original image at random

        image = Image.open(self.path_prefix + self.data["image_path"].iloc[i])
        image_aug = TF.to_tensor(smart_hue_adjust(image, hue_shift))

        if self.resize:
            resize = Resize(self.resize)
            image_aug = resize(image_aug)

        palette = Image.fromarray(np.load(self.path_prefix + self.data["palette_path"].iloc[i]).astype(np.uint8))
        palette_aug = TF.to_tensor(smart_hue_adjust(palette, hue_shift))

        return image_aug, palette_aug

    def __len__(self):
        """

        :return:
        """
        return len(self.data) * self.multiplier
