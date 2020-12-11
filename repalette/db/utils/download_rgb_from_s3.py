import boto3
from botocore.exceptions import ClientError
import logging
import os
import tempfile
import zipfile

from repalette.constants import (
    BASE_DATA_DIR,
    RGB_DATABASE_PATH,
    S3_BUCKET_NAME,
    S3_RGB_IMAGES_PATH,
    S3_RGB_DATABASE_PATH,
    ROOT_DIR,
)


def download_from_s3():
    s3_client = boto3.client("s3")

    os.makedirs(BASE_DATA_DIR, exist_ok=True)

    tmp_file_name = os.path.join(BASE_DATA_DIR, "rgb_tmp.zip")

    try:
        print(
            f"Downloading from s3://{S3_BUCKET_NAME}/{S3_RGB_IMAGES_PATH} to temporary archive {tmp_file_name}"
        )
        s3_client.download_file(S3_BUCKET_NAME, S3_RGB_DATABASE_PATH, RGB_DATABASE_PATH)

        s3_client.download_file(S3_BUCKET_NAME, S3_RGB_IMAGES_PATH, tmp_file_name)
        print(f"Extracting archive to {BASE_DATA_DIR}")
        with zipfile.ZipFile(tmp_file_name, "r") as zip_ref:
            zip_ref.extractall(ROOT_DIR)

    except ClientError as e:
        logging.error(e)

    print(f"Removing temporary archive {tmp_file_name}")
    os.remove(f"{tmp_file_name}")


if __name__ == "__main__":
    download_from_s3()
