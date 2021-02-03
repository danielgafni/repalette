import numpy as np
from PIL import Image
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from repalette.constants import RGB_DATABASE_PATH
from repalette.datasets import AbstractQueryDataset
from repalette.db.rgb import RGBImage


class RGBDataset(AbstractQueryDataset):
    """
    Dataset of RGB images.
    `repalette/utils/build_rgb.py` must be run before using this dataset
    """

    def __init__(self, query=None, random_seed=None):
        if query is None:
            engine = create_engine(f"sqlite:///{RGB_DATABASE_PATH}")
            # create a configured "Session" class
            Session = sessionmaker(bind=engine)
            session = Session()
            query = session.query(RGBImage).all()
            session.close()

        super().__init__(query=query, random_seed=random_seed)

    def _getitem(self, index):
        rgb_image = self.query[index]
        image = np.array(Image.open(rgb_image.path).convert("RGB"))

        return (
            image,
            rgb_image.palette,
        ), rgb_image
