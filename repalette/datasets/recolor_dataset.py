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
        hue_shift = (index % self.multiplier - (self.multiplier - 1) / 2) / (
            self.multiplier - 1
        )
        i = (
            index // self.multiplier
        )  # actual image index (from design-seeds-data directory)

        rgb_image = self.query[i]

        image = Image.open(rgb_image.path)

        if self.resize:
            resize = Resize(self.resize)
            image = resize(image)

        image_aug = TF.to_tensor(
            smart_hue_adjust(image, hue_shift),
        )

        palette = Image.fromarray(rgb_image.palette
                                  )
        palette_aug = TF.to_tensor(smart_hue_adjust(palette, hue_shift))

        return image_aug, palette_aug

    def __len__(self):
        """
        :return:
        """
        return self.query.count() * self.multiplier
