from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as TF
from torchvision.transforms import Resize
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import random

from repalette.constants import RGB_DATABASE_PATH
from repalette.utils.transforms import smart_hue_adjust
from repalette.db.rgb import RGBImage


class RecolorDataset(Dataset):
    def __init__(
        self,
        multiplier: int,
        query=None,
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

        if query is None:
            engine = create_engine(f"sqlite:///{RGB_DATABASE_PATH}")
            # create a configured "Session" class
            Session = sessionmaker(bind=engine)
            session = Session()
            self.query = session.query(RGBImage).all()
            session.close()
        else:
            self.query = query

        if shuffle:
            random.shuffle(self.query)

    def __getitem__(self, index):
        """
        :param index: index of item to get from the dataset
        :return: image of shape [3, self.resize] and palette of shape [3, 1, 6]
        """
        hue_shift = (index % self.multiplier - (self.multiplier - 1) / 2) / (self.multiplier - 1)
        i = index // self.multiplier  # actual image index (from design-seeds-data directory)

        rgb_image = self.query[i]

        image = Image.open(rgb_image.path)

        if self.resize:
            resize = Resize(self.resize)
            image = resize(image)

        image_aug = TF.to_tensor(smart_hue_adjust(image, hue_shift))

        palette = Image.fromarray(rgb_image.palette)
        palette_aug = TF.to_tensor(smart_hue_adjust(palette, hue_shift))

        return image_aug, palette_aug

    def __len__(self):
        return len(self.query) * self.multiplier

    def split(self, test_size=0.2, shuffle=True):
        query = self.query

        if shuffle:
            random.shuffle(query)

        train_query = query[: int(len(query) * (1 - test_size))]
        test_query = query[int(len(query) * (1 - test_size)) :]

        train = RecolorDataset(query=train_query)
        test = RecolorDataset(query=test_query)

        return train, test
