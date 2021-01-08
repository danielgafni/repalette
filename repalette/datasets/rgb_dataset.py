from torch.utils.data import Dataset
from PIL import Image
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import numpy as np
import random

from repalette.constants import RGB_DATABASE_PATH
from repalette.db.rgb import RGBImage


class RGBDataset(Dataset):
    """
    Dataset of RGB images.
    `repalette/utils/build_rgb.py` must be run before using this dataset
    """

    def __init__(self, query=None):
        if query is None:
            engine = create_engine(f"sqlite:///{RGB_DATABASE_PATH}")
            # create a configured "Session" class
            Session = sessionmaker(bind=engine)
            session = Session()
            self.query = session.query(RGBImage).all()
            session.close()
        else:
            self.query = query

    def __getitem__(self, index):
        rgb_image = self.query[index]

        if not rgb_image:
            raise IndexError

        image = np.array(Image.open(rgb_image.path).convert("RGB"))

        return (
            image,
            rgb_image.palette,
        ), rgb_image

    def __len__(self):
        return len(self.query)

    def split(self, test_size=0.2, shuffle=True):
        query = self.query

        if shuffle:
            random.shuffle(query)

        train_query = query[: int(len(query) * (1 - test_size))]
        test_query = query[int(len(query) * (1 - test_size)) :]

        train = RGBDataset(query=train_query)
        test = RGBDataset(query=test_query)

        return train, test
