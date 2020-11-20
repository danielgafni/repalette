import os
import torch

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))[:-9]
BASE_DATA_DIR = os.path.join(ROOT_DIR, "data")
RAW_DATA_DIR = os.path.join(BASE_DATA_DIR, "raw")
RGB_IMAGES_DIR = os.path.join(BASE_DATA_DIR, "rgb")
LAB_DATA_DIR = os.path.join(BASE_DATA_DIR, "lab")
LAB_IMAGES_DIR = os.path.join(LAB_DATA_DIR, "images")
LAB_PALETTES_DIR = os.path.join(LAB_DATA_DIR, "palettes")
MODELS_DIR = os.path.join(BASE_DATA_DIR, "models")
MODEL_SAVES_DIR = os.path.join(BASE_DATA_DIR, "saves")
PL_LOGS_DIR = os.path.join(BASE_DATA_DIR, "pl_logs")
DATABASE_PATH = os.path.join(BASE_DATA_DIR, "sqlite.db")
DEFAULT_DATABASE = f"sqlite:///{DATABASE_PATH}"
LIGHTNINGS_LOGS_DIR = "lightning_logs"

IMAGE_SIZE = (432, 288)
DEFAULT_LR = 0.0002
DEFAULT_BETA_1 = 0.5
DEFAULT_BETA_2 = 0.999

L_RANGE = (0, 100)
A_RANGE = (-86.185, 98.254)
B_RANGE = (-107.863, 94.482)
# LAB_RANGE = torch.as_tensor([L_RANGE, A_RANGE, B_RANGE], dtype=torch.float)
# NORM_RANGE = torch.as_tensor([[-1, 1]] * 3, dtype=torch.float)
