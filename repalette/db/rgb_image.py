import datetime
from sqlalchemy import Column, Integer, String, DateTime
import numpy as np

from repalette.db import Base


class RGBImage(Base):
    __tablename__ = "rgb_images"
    id = Column("id", Integer, primary_key=True)
    path = Column("path", String)
    url = Column("url", String, unique=True)
    name = Column("name", String, default="")
    height = Column("height", Integer, default=0)
    width = Column("width", Integer, default=0)
    created_at = Column("created_at", DateTime, default=datetime.datetime.utcnow)

    def __init__(self, path, np_palette, url="", name="", height=0, width=0):
        self.path = path
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
                setattr(self, f"color_{i}_{c}", int(value))

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


# add 18 columns for 6 RGB colors - don't want to do it manually :)
for i in range(6):
    for c in "rgb":
        setattr(
            RGBImage, f"color_{i}_{c}", Column(f"color_{i}_{c}", Integer, default=0)
        )
