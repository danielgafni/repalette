from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as TF
from torchvision.transforms import Resize
import numpy as np
from pandas import DataFrame
from repalette.constants import ROOT_DIR
import torch


class NaiveRecolorDataset(Dataset):
    def __init__(self, data: DataFrame, multiplier: int, path_prefix: str = ROOT_DIR, resize=None):
        """
        Dataset constructor.
        :param data: DataFrame containing columns `image_path` and `palette_path`
        :param multiplier: multiplier for color augmentation
        :param path_prefix: full path prefix to add before relative paths in data
        :param resize: size to which the image will be resized with `torhvision.trainsforms.Resize`
        """
        self.multiplier = multiplier
        self.path_prefix = path_prefix
        self.resize = resize
        self.data = data

    def __getitem__(self, index):
        """
        :param index: index of item to get from the dataset
        :return: returns image of shape [3, self.resize] and palette of shape [3, 1, 6]
        """
        hue_shift = (index % self.multiplier - (self.multiplier - 1) / 2) / (self.multiplier - 1)
        i = index // self.multiplier  # actual image index (from design-seeds-data directory)

        image = Image.open(self.path_prefix + self.data["image_path"].iloc[i])
        image = TF.adjust_hue(image, hue_shift)

        if self.resize:
            resize = Resize(self.resize)
            image = resize(image)

        palette = Image.fromarray(np.load(self.path_prefix + self.data["palette_path"].iloc[i]).astype(np.uint8))
        palette = TF.adjust_hue(palette, hue_shift)
        palette_converted = np.array(palette.convert("HSV"))
        hue_sort_args = torch.tensor(palette_converted).squeeze(0)[:, 0].argsort()

        # palette_unsorted = torch.tensor(np.array(palette)).float().permute(2, 0, 1) / 255

        image = torch.tensor(np.array(image)).float().permute(2, 0, 1) / 255
        palette_sorted = (torch.tensor(np.array(palette)).float().permute(2, 0, 1) / 255)[:, :, hue_sort_args]

        return image, palette_sorted

    def __len__(self):
        """

        :return:
        """
        return len(self.data) * self.multiplier

