from torch.utils.data import Dataset
from PIL import Image
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import random

from repalette.constants import DATABASE_PATH
from repalette.utils.models import RawImage


class RawDataset(Dataset):
    """
    Dataset of images downloaded from https://www.design-seeds.com/blog/.
    `repalette/utils/download_data.py` must be run before using this dataset
    """

    def __init__(self, query=None):
        if query is None:
            engine = create_engine(f"sqlite:///{DATABASE_PATH}")
            # create a configured "Session" class
            Session = sessionmaker(bind=engine)
            session = Session()
            self.query = session.query(RawImage).all()
            session.close()
        else:
            self.query = query

    def __getitem__(self, index):
        raw_image = self.query[index]

        if not raw_image:
            raise IndexError

        image = Image.open(raw_image.path).convert("RGB")

        return (image, raw_image.palette), raw_image

    def __len__(self):
        return len(self.query)

    def split(self, test_size=0.2, shuffle=True):
        query = self.query

        if shuffle:
            random.shuffle(query)

        train_query = query[:int(len(query) * (1 - test_size))]
        test_query = query[int(len(query) * (1 - test_size)):]

        train = RawDataset(query=train_query)
        test = RawDataset(query=test_query)

        return train, test
