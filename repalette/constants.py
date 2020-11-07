import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))[:-9]
MODELS_DIR = ROOT_DIR + "repalette/models/pytorch_models_checkpoints/"
IMAGE_SIZE = (432, 288)
LIGHTNINGS_LOGS_DIR = "lightning_logs"
DEFAULT_LR = 0.0002
DEFAULT_BETAS = (0.5, 0.999)
