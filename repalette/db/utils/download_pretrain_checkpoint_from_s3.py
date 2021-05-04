import logging
import os

import boto3
from botocore.exceptions import ClientError

from repalette.constants import (
    BASE_DATA_DIR,
    PRETRAINED_MODEL_CHECKPOINT_PATH,
    S3_BUCKET_NAME,
    S3_MODEL_CHECKPOINTS_RELATIVE_DIR,
    S3_PRETRAINED_MODEL_CHECKPOINT_PATH,
)


def download_from_s3():
    s3_client = boto3.client("s3")

    os.makedirs(BASE_DATA_DIR, exist_ok=True)

    try:
        print(
            f"Downloading pretrained model checkpoint from {os.path.join(S3_BUCKET_NAME, S3_MODEL_CHECKPOINTS_RELATIVE_DIR, S3_PRETRAINED_MODEL_CHECKPOINT_PATH)} to {PRETRAINED_MODEL_CHECKPOINT_PATH}"
        )
        s3_client.download_file(
            S3_BUCKET_NAME,
            S3_PRETRAINED_MODEL_CHECKPOINT_PATH,
            PRETRAINED_MODEL_CHECKPOINT_PATH,
        )

    except ClientError as e:
        logging.error(e)


if __name__ == "__main__":
    download_from_s3()
