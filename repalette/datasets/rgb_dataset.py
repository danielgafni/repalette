from torch.utils.data import Dataset
from PIL import Image
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import numpy as np

from repalette.constants import DATABASE_PATH
from repalette.utils.models import RGBImage


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

        session.close()

    def __getitem__(self, index):
        rgb_image = self.query.get(index + 1)

        if not rgb_image:
            raise IndexError

        image = np.array(Image.open(rgb_image.path).convert("RGB"))

        return (image, rgb_image.palette), rgb_image

    def __len__(self):
        return self.query.count()
