from sqlalchemy import MetaData
from sqlalchemy.ext.declarative import declarative_base

meta = MetaData()

Base = declarative_base()

from .raw_image import RawImage  # import order matters!
