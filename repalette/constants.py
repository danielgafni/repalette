import os
from dotenv import load_dotenv

load_dotenv()

AWS_DEFAULT_REGION = os.getenv("AWS_DEFAULT_REGION")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")
S3_BUCKET_PATH = f"s3://{S3_BUCKET_NAME}/"

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_DATA_DIR = os.path.join(ROOT_DIR, "data")
AWS_DIR = os.path.join(BASE_DATA_DIR, "aws")
AWS_ACCESS_KEY_PATH = os.path.join(AWS_DIR, "access_key.csv")
ENV_PATH = os.path.join(ROOT_DIR, ".env")
RAW_DATA_DIR = os.path.join(BASE_DATA_DIR, "raw")
RGB_IMAGES_DIR = os.path.join(BASE_DATA_DIR, "rgb")
LAB_DATA_DIR = os.path.join(BASE_DATA_DIR, "lab")
LAB_IMAGES_DIR = os.path.join(LAB_DATA_DIR, "images")
LAB_PALETTES_DIR = os.path.join(LAB_DATA_DIR, "palettes")
MODELS_DIR = os.path.join(BASE_DATA_DIR, "models")
MODEL_CHECKPOINTS_DIR = os.path.join(BASE_DATA_DIR, "saves")
LIGHTNING_LOGS_DIR = os.path.join(BASE_DATA_DIR, "lightning-logs")
DATABASE_PATH = os.path.join(BASE_DATA_DIR, "sqlite.db")
DEFAULT_DATABASE = f"sqlite:///{DATABASE_PATH}"
S3_LIGHTNING_LOGS_DIR = os.path.join(S3_BUCKET_PATH, "lightning-logs")
S3_MODEL_CHECKPOINTS_DIR = os.path.join(S3_BUCKET_PATH, "model-checkpoints")

IMAGE_SIZE = (432, 288)
DEFAULT_LR = 0.0002
DEFAULT_BETA_1 = 0.5
DEFAULT_BETA_2 = 0.999
DEFAULT_LAMBDA_MSE_LOSS = 10

L_RANGE = (0, 100)
A_RANGE = (-86.185, 98.254)
B_RANGE = (-107.863, 94.482)
