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
            self.query = session.query(RawImage)
            session.close()
        else:
            self.query = query

    def __getitem__(self, index):
        raw_image = self.query.get(index + 1)

        if not raw_image:
            raise IndexError

        image = Image.open(raw_image.path).convert("RGB")

        return (image, raw_image.palette), raw_image

    def __len__(self):
        return self.query.count()

    def split(self, test_size=0.2, shuffle=True):
        all_indices = list(range(1, self.query.count() + 1))

        if shuffle:
            random.shuffle(all_indices)

        train_indices = all_indices[:int(len(all_indices) * (1 - test_size))]
        test_indices = all_indices[int(len(all_indices) * (1 - test_size)):]

        train_query = self.query.filter(RawImage.id.in_(train_indices))
        test_query = self.query.filter(RawImage.id.in_(test_indices))

        train = RawDataset(query=train_query)
        test = RawDataset(query=test_query)

        return train, test
