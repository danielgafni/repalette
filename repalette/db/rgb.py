import datetime
from sqlalchemy import (
    Column,
    Integer,
    String,
    DateTime,
    MetaData,
)
from sqlalchemy.ext.declarative import declarative_base, DeclarativeMeta
import numpy as np
import os

from repalette.constants import RGB_IMAGES_DIR

rgb_meta = MetaData()

RGBBase: DeclarativeMeta = declarative_base()  # use DeclarativeMeta to calm down mypy


class RGBImage(RGBBase):
    __tablename__ = "rgb_images"
    id = Column("id", Integer, primary_key=True)
    url = Column("url", String, unique=True)
    name = Column("name", String, default="")
    height = Column("height", Integer, default=0)
    width = Column("width", Integer, default=0)
    created_at = Column(
        "created_at",
        DateTime,
        default=datetime.datetime.utcnow,
    )

    def __init__(
        self,
        np_palette,
        url="",
        name="",
        height=0,
        width=0,
    ):
        self.url = url
        self.name = name
        self.height = height
        self.width = width

        self.set_palette(np_palette)

    def set_palette(self, np_palette):
        """
        Sets colors
        :param np_palette: numpy rgb palette with shape of [1, 6, 3]
        """
        for i, color in enumerate(np_palette.reshape(6, 3)):
            for value, c in zip(color, "rgb"):
                setattr(
                    self,
                    f"color_{i}_{c}",
                    int(value),
                )

    @property
    def palette(self):
        """
        Returns numpy rgb palette with shape of [1, 6, 3]
        """
        palette = []
        for i in range(6):
            color = []
            for c in "rgb":
                color.append(getattr(self, f"color_{i}_{c}"))
            palette.append(color)
        return np.array(palette).reshape(1, 6, 3).astype(np.uint8)

    @property
    def path(self):
        path = os.path.join(RGB_IMAGES_DIR, self.name)
        return path


# add 18 columns for 6 RGB colors - don't want to do it manually :)
for i in range(6):
    for c in "rgb":
        setattr(
            RGBImage,
            f"color_{i}_{c}",
            Column(
                f"color_{i}_{c}",
                Integer,
                default=0,
            ),
        )
