from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as TF
import torch
from torchvision import transforms
import numpy as np
from pandas import DataFrame
from itertools import permutations
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from repalette.constants import ROOT_DIR, IMAGE_SIZE, DATABASE_PATH
from repalette.utils.color import smart_hue_adjust, PairHueAdjust
from repalette.utils.models import RawImage, RGBImage


class RecolorDataset(Dataset):
    def __init__(
            self,
            data: DataFrame,
            multiplier: int,
            path_prefix: str = ROOT_DIR,
            resize=IMAGE_SIZE,
    ):
        """
        Dataset constructor.
        :param data: DataFrame containing columns `image_path` and `palette_path`
        :param multiplier: an odd multiplier for color augmentation
        :param path_prefix: full path prefix to add before relative paths in data
        :param resize: size to which the image will be resized with `torсhvision.transforms.Resize`
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
        hue_shift = (index % self.multiplier - (self.multiplier - 1) / 2) / (
                self.multiplier - 1
        )
        i = (
                index // self.multiplier
        )  # actual image index (from design-seeds-data directory)

        image = Image.open(self.path_prefix + self.data["image_path"].iloc[i])
        if self.resize:
            resize = transforms.Resize(self.resize)
            image = resize(image)
        image_aug = TF.to_tensor(
            smart_hue_adjust(image, hue_shift),
        )

        palette = Image.fromarray(
            np.load(self.path_prefix + self.data["palette_path"].iloc[i]).astype(
                np.uint8
            )
        )
        palette_aug = TF.to_tensor(smart_hue_adjust(palette, hue_shift))

        return image_aug, palette_aug

    def __len__(self):
        """
        :return:
        """
        return len(self.data) * self.multiplier


# class PairRecolorDataset(Dataset):
#     def __init__(
#         self,
#         data: DataFrame,
#         multiplier: int,
#         path_prefix: str = ROOT_DIR,
#         resize: tuple = IMAGE_SIZE,
#         shuffle_palette=True,
#     ):
#         """
#         Dataset constructor.
#         :param data: DataFrame containing columns `image_path` and `palette_path`
#         :param multiplier: an odd multiplier for color augmentation
#         :param path_prefix: full path prefix to add before relative paths in data
#         :param resize: size to which the image will be resized with
#         `torhvision.trainsforms.Resize`
#         :param shuffle_palette: if to shuffle output palettes
#         """
#         # if multiplier % 2 == 0:
#         #     raise ValueError("Multiplier must be odd.")
#         self.multiplier = multiplier
#         self.path_prefix = path_prefix
#         self.resize = resize
#         self.data = data
#         self.shuffle_palette = shuffle_palette
#
#         hue_variants = np.linspace(-0.5, 0.5, self.multiplier)
#         self.hue_pairs = [perm for perm in permutations(hue_variants, 2)]
#         self.n_pairs = len(self.hue_pairs)
#
#     def __getitem__(self, index):
#         """
#         :param index: index of item to get from the dataset
#         :return: image of shape [3, self.resize] and palette of shape [3, 1, 6]
#         """
#         pair_index = index % self.n_pairs
#         hue_shift_1, hue_shift_2 = self.hue_pairs[pair_index]
#         i = (
#             index // self.n_pairs
#         )  # actual image index (from design-seeds-data directory)
#
#         image = Image.open(self.path_prefix + str(self.data["image_path"].iloc[i]))
#         if self.resize:
#             resize = transforms.Resize(self.resize)
#             image = resize(image)
#         image_aug_1 = TF.to_tensor(smart_hue_adjust(image, hue_shift_1)).to(torch.float)
#         image_aug_2 = TF.to_tensor(smart_hue_adjust(image, hue_shift_2)).to(torch.float)
#
#         palette = Image.fromarray(
#             np.load(self.path_prefix + self.data["palette_path"].iloc[i]).astype(
#                 np.uint8
#             )
#         )
#         palette_aug_1 = TF.to_tensor(smart_hue_adjust(palette, hue_shift_1)).to(
#             torch.float
#         )
#         palette_aug_2 = TF.to_tensor(smart_hue_adjust(palette, hue_shift_2)).to(
#             torch.float
#         )
#
#         if self.shuffle_palette:
#             palette_aug_1 = palette_aug_1[:, :, torch.randperm(6)]
#             palette_aug_2 = palette_aug_2[:, :, torch.randperm(6)]
#
#         return (image_aug_1, palette_aug_1), (image_aug_2, palette_aug_2)
#
#     def __len__(self):
#         """
#         :return:
#         """
#         return len(self.data) * self.n_pairs
#
#     def shuffle(self, set_shuffle=True):
#         """
#         Shuffles data.
#         :param set_shuffle: set data to shuffled or unshuffled state
#         """
#         if set_shuffle:
#             self.data = self.data.sample(frac=1)
#
#         else:
#             self.data = self.data.reindex(list(range(len(self.data))))
#
#         return self


class ShuffleDataLoader(torch.utils.data.DataLoader):
    def __init__(self, *args, **kwargs):
        super(ShuffleDataLoader, self).__init__(*args, **kwargs)

    def shuffle(self, set_shuffle=True):
        self.dataset.shuffle(set_shuffle)
        return self


class RawDataset(Dataset):
    """
    Dataset of images downloaded from https://www.design-seeds.com/blog/.
    `repalette/utils/download_data.py` must be run before using this dataset
    """

    def __init__(self):
        engine = create_engine(f"sqlite:///{DATABASE_PATH}")
        # create a configured "Session" class
        Session = sessionmaker(bind=engine)
        session = Session()

        self.query = session.query(RawImage)
        self.length = self.query.count()

        session.close()

    def __getitem__(self, index):
        raw_image = self.query.get(index + 1)

        if not raw_image:
            raise IndexError

        image = Image.open(raw_image.path).convert("RGB")

        return (image, raw_image.palette), raw_image

    def __len__(self):
        return self.length


class RGBDataset(Dataset):
    """
    Dataset of RGB images.
    `repalette/utils/build_rgb.py` must be run before using this dataset
    """

    def __init__(self):
        engine = create_engine(f"sqlite:///{DATABASE_PATH}")
        # create a configured "Session" class
        Session = sessionmaker(bind=engine)
        session = Session()

        self.query = session.query(RGBImage)
        self.length = self.query.count()

        session.close()

    def __getitem__(self, index):
        rgb_image = self.query.get(index + 1)

        if not rgb_image:
            raise IndexError

        image = Image.open(rgb_image.path).convert("RGB")

        return (image, rgb_image.palette), rgb_image

    def __len__(self):
        return self.length


class PairRecolorDataset(Dataset):
    def __init__(
            self,
            data: DataFrame,
            multiplier: int,
            path_prefix: str = ROOT_DIR,
            shuffle_palette=True,
            transform=None
    ):
        """
        Dataset constructor.
        :param data: DataFrame containing columns `image_path` and `palette_path`
        :param multiplier: an odd multiplier for color augmentation
        :param path_prefix: full path prefix to add before relative paths in data
        :param shuffle_palette: if to shuffle output palettes
        :param transform: optional transform to be applied on a sample
        """
        # if multiplier % 2 == 0:
        #     raise ValueError("Multiplier must be odd.")
        self.multiplier = multiplier
        self.path_prefix = path_prefix
        self.data = data
        self.shuffle_palette = shuffle_palette

        hue_variants = np.linspace(-0.5, 0.5, self.multiplier)
        self.hue_pairs = [perm for perm in permutations(hue_variants, 2)]
        self.n_pairs = len(self.hue_pairs)

        self.transform = transform
        if transform is None:
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(180),
            ])

    def __getitem__(self, index):
        """
        :param index: index of item to get from the dataset
        :return: image of shape [3, self.resize] and palette of shape [3, 1, 6]
        """
        pair_index = index % self.n_pairs
        hue_shift_first, hue_shift_second = self.hue_pairs[pair_index]
        i = index // self.n_pairs  # actual image index (from design-seeds-data directory)

        img_transform = PairHueAdjust(self.transform)
        palette_transform = PairHueAdjust()

        image = Image.open(self.path_prefix + str(self.data["image_path"].iloc[i]))
        img_aug_first, img_aug_second = img_transform(image, hue_shift_first, hue_shift_second)

        palette = Image.fromarray(
            np.load(self.path_prefix + self.data["palette_path"].iloc[i]).astype(
                np.uint8
            )
        )
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
        return len(self.data) * self.n_pairs

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
