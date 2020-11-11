from sqlalchemy import MetaData
from sqlalchemy.ext.declarative import declarative_base

meta = MetaData()

Base = declarative_base()

from .raw_image import RawImage  # import order matters!
from .rgb_image import RGBImage
from .lab_tensors import LABTensor


def image_url_to_name(image_url):
    return image_url.split("/")[-1]
