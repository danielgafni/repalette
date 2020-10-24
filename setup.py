import setuptools
from distutils.core import setup
import os


setup(
   name='repalette',
   version='0.0.0',
   description='A recoloring tool usable with any color palette.',
   setup_requires=['wheel', 'colorgram.py'],
)


if not os.path.exists("repalette/models/pytorch_models_checkpoints"):
    os.mkdir("repalette/models/pytorch_models_checkpoints")
