import torch
from torch.utils.data import Dataset
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from repalette.constants import DATABASE_PATH
from repalette.utils.models import LABTensor


class LABDataset(Dataset):
    """
    Dataset of images downloaded from https://www.design-seeds.com/blog/.
    `repalette/utils/download_data.py` must be run before using this dataset
    """

    def __init__(self):
        engine = create_engine(f"sqlite:///{DATABASE_PATH}")
        # create a configured "Session" class
        Session = sessionmaker(bind=engine)
        session = Session()

        self.query = session.query(LABTensor)
        self.length = self.query.count()

        self.images = [torch.load(lab_tensor.image_path) for lab_tensor in self.query]
        self.palettes = [
            torch.load(lab_tensor.palette_path) for lab_tensor in self.query
        ]

        session.close()

    def __getitem__(self, index):
        lab_tensor = self.query.get(index + 1)

        if not lab_tensor:
            raise IndexError

        image = self.images[index]
        palette = self.palettes[index]

        return (image, palette), lab_tensor

    def __len__(self):
        return self.length
