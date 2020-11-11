from torch.utils.data import Dataset
from PIL import Image
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from repalette.constants import DATABASE_PATH
from repalette.utils.models import RawImage


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
