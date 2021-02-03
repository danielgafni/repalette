import logging
import os

import boto3
from botocore.exceptions import ClientError

from repalette.constants import (
    BASE_DATA_DIR,
    PRETRAINED_MODEL_CHECKPOINT_PATH,
    S3_BUCKET_NAME,
    S3_PRETRAINED_MODEL_CHECKPOINT_PATH,
)


def download_from_s3():
    s3_client = boto3.client("s3")

    os.makedirs(BASE_DATA_DIR, exist_ok=True)

    tmp_file_name = os.path.join(BASE_DATA_DIR, "rgb_tmp.zip")

    try:
        print(
            f"Downloading from s3://{S3_BUCKET_NAME}/{S3_PRETRAINED_MODEL_CHECKPOINT_PATH} to {PRETRAINED_MODEL_CHECKPOINT_PATH}"
        )
        s3_client.download_file(
            S3_BUCKET_NAME,
            S3_PRETRAINED_MODEL_CHECKPOINT_PATH,
            PRETRAINED_MODEL_CHECKPOINT_PATH,
        )

    except ClientError as e:
        logging.error(e)

    print(f"Removing temporary archive {tmp_file_name}")
    os.remove(f"{tmp_file_name}")


if __name__ == "__main__":
    download_from_s3()
