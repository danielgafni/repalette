import os
from dotenv import load_dotenv

load_dotenv()

# AWS
AWS_DEFAULT_REGION = os.getenv("AWS_DEFAULT_REGION")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")
S3_BUCKET_PATH = f"s3://{S3_BUCKET_NAME}/"

S3_BASE_DATA_DIR = "data/"
S3_RAW_DATA_PATH = os.path.join(S3_BASE_DATA_DIR, "raw.zip")
S3_RGB_IMAGES_PATH = os.path.join(S3_BASE_DATA_DIR, "rgb.zip")

S3_RAW_DATABASE_PATH = os.path.join(S3_BASE_DATA_DIR, "raw.sqlite")
S3_RGB_DATABASE_PATH = os.path.join(S3_BASE_DATA_DIR, "rgb.sqlite")

S3_LIGHTNING_LOGS_RELATIVE_DIR = "lightning-logs"
S3_LIGHTNING_LOGS_DIR = os.path.join(S3_BUCKET_PATH, S3_LIGHTNING_LOGS_RELATIVE_DIR)
S3_MODEL_CHECKPOINTS_RELATIVE_DIR = "model-checkpoints"
S3_MODEL_CHECKPOINTS_DIR = os.path.join(
    S3_BUCKET_PATH, S3_MODEL_CHECKPOINTS_RELATIVE_DIR
)

# data
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_DATA_DIR = os.path.join(ROOT_DIR, "data")
RAW_DATA_DIR = os.path.join(BASE_DATA_DIR, "raw")
RGB_IMAGES_DIR = os.path.join(BASE_DATA_DIR, "rgb")

RAW_DATABASE_PATH = os.path.join(BASE_DATA_DIR, "raw.sqlite")
RGB_DATABASE_PATH = os.path.join(BASE_DATA_DIR, "rgb.sqlite")
DEFAULT_RAW_DATABASE = f"sqlite:///{RAW_DATABASE_PATH}"
DEFAULT_RGB_DATABASE = f"sqlite:///{RGB_DATABASE_PATH}"

LIGHTNING_LOGS_DIR = os.path.join(BASE_DATA_DIR, "lightning-logs")
MODEL_CHECKPOINTS_DIR = os.path.join(BASE_DATA_DIR, "model-checkpoints")
MODELS_DIR = os.path.join(BASE_DATA_DIR, "models")

# transforms
IMAGE_SIZE = (432, 288)
L_RANGE = (0, 100)
A_RANGE = (-86.185, 98.254)
B_RANGE = (-107.863, 94.482)

# pretraining hyperparameters
DEFAULT_PRETRAIN_LR = 2e-5
DEFAULT_PRETRAIN_BETA_1 = 0.9
DEFAULT_PRETRAIN_BETA_2 = 0.999
DEFAULT_PRETRAIN_WEIGHT_DECAY = 0.01

# adversarial hyperparameters
DEFAULT_ADVERSARIAL_LR = 0.0002
DEFAULT_ADVERSARIAL_BETA_1 = 0.5
DEFAULT_ADVERSARIAL_BETA_2 = 0.999
DEFAULT_ADVERSARIAL_LAMBDA_MSE_LOSS = 10
