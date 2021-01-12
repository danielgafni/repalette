from PIL import Image
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from repalette.constants import RAW_DATABASE_PATH
from repalette.db.raw import RawImage

from repalette.datasets import AbstractQueryDataset


class RawDataset(AbstractQueryDataset):
    """
    Dataset of images downloaded from https://www.design-seeds.com/blog/.
    `repalette/utils/download_raw.py` must be run before using this dataset
    """

    def __init__(self, query=None, random_seed=None):
        if query is None:
            engine = create_engine(f"sqlite:///{RAW_DATABASE_PATH}")
            # create a configured "Session" class
            Session = sessionmaker(bind=engine)
            session = Session()
            query = session.query(RawImage).all()
            session.close()

        super().__init__(query=query, random_seed=random_seed)

    def _getitem(self, index):
        raw_image = self.query[index]
        image = Image.open(raw_image.path).convert("RGB")

        return (
            image,
            raw_image.palette,
        ), raw_image
