import datetime
import os

from sqlalchemy import Column, DateTime, Integer, MetaData, String
from sqlalchemy.ext.declarative import DeclarativeMeta, declarative_base

from repalette.constants import RAW_DATA_DIR

raw_meta = MetaData()

RAWBase: DeclarativeMeta = declarative_base()  # use DeclarativeMeta to calm down mypy


class RawImage(RAWBase):
    __tablename__ = "raw_images"
    id = Column("id", Integer, primary_key=True)
    url = Column("url", String, unique=True)
    name = Column("name", String, default="")
    height = Column("height", Integer, default=0)
    width = Column("width", Integer, default=0)
    palette_0 = Column("palette_0", String, default="")
    palette_1 = Column("palette_1", String, default="")
    palette_2 = Column("palette_2", String, default="")
    palette_3 = Column("palette_3", String, default="")
    palette_4 = Column("palette_4", String, default="")
    palette_5 = Column("palette_5", String, default="")
    created_at = Column(
        "created_at",
        DateTime,
        default=datetime.datetime.utcnow,
    )

    def __init__(
        self,
        palette,
        url="",
        name="",
        height=0,
        width=0,
    ):
        self.url = url
        self.name = name
        self.height = height
        self.width = width

        self.set_palette(palette)

    def set_palette(self, palette):
        """
        Sets palette colors
        :param palette: list of colors in HEX
        """
        for i, color in enumerate(palette):
            setattr(self, f"palette_{i}", color)

    @property
    def palette(self):
        """
        Returns palette colors
        """
        return [getattr(self, f"palette_{i}") for i in range(6)]

    @property
    def path(self):
        path = os.path.join(RAW_DATA_DIR, self.name)
        return path
