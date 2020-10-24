import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))[:-9]
MODELS_DIR = ROOT_DIR + "repalette/models/pytorch_models_checkpoints/"
IMAGE_SIZE = (432, 288)
LR = 0.0002
BETAS = (0.5, 0.999)
