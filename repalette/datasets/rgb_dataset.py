from torch.utils.data import Dataset
from PIL import Image
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import numpy as np
import random

from repalette.constants import DATABASE_PATH
from repalette.utils.models import RGBImage


class RGBDataset(Dataset):
    """
    Dataset of RGB images.
    `repalette/utils/build_rgb.py` must be run before using this dataset
    """

    def __init__(self, query=None):
        if query is not None:
            engine = create_engine(f"sqlite:///{DATABASE_PATH}")
            # create a configured "Session" class
            Session = sessionmaker(bind=engine)
            session = Session()
            self.query = session.query(RGBImage)
            session.close()
        else:
            self.query = query

    def __getitem__(self, index):
        rgb_image = self.query.get(index + 1)

        if not rgb_image:
            raise IndexError

        image = np.array(Image.open(rgb_image.path).convert("RGB"))

        return (image, rgb_image.palette), rgb_image

    def __len__(self):
        return self.query.count()

    def split(self, test_size=0.2, shuffle=True):
        all_indices = list(range(1, self.query.count() + 1))

        if shuffle:
            random.shuffle(all_indices)

        train_indices = all_indices[:int(len(all_indices) * (1 - test_size))]
        test_indices = all_indices[int(len(all_indices) * (1 - test_size)):]

        train_query = self.query.filter(RGBImage.id.in_(train_indices))
        test_query = self.query.filter(RGBImage.id.in_(test_indices))

        train = RGBDataset(query=train_query)
        test = RGBDataset(query=test_query)

        return train, test
