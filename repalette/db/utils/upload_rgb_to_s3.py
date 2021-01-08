import logging
import shutil
import os
import boto3
from botocore.exceptions import ClientError

from repalette.constants import (
    S3_RGB_DATABASE_PATH,
    S3_BUCKET_NAME,
    S3_RGB_IMAGES_PATH,
    BASE_DATA_DIR,
    RGB_IMAGES_DIR,
    RGB_DATABASE_PATH,
    ROOT_DIR,
)


def upload_to_s3():
    # TODO: use repalette.utils.aws.upload_to_s3
    s3_client = boto3.client("s3")
    tmp_file_name = os.path.join(BASE_DATA_DIR, "rgb_tmp")
    print(f"Creating temporary archive {tmp_file_name}.zip")
    shutil.make_archive(
        tmp_file_name,
        "zip",
        root_dir=ROOT_DIR,
        base_dir=os.path.relpath(RGB_IMAGES_DIR, ROOT_DIR),
    )

    try:
        print(f"Uploading {tmp_file_name}.zip to s3://{S3_BUCKET_NAME}/{S3_RGB_DATABASE_PATH}")
        s3_client.upload_file(f"{tmp_file_name}.zip", S3_BUCKET_NAME, S3_RGB_IMAGES_PATH)
        s3_client.upload_file(RGB_DATABASE_PATH, S3_BUCKET_NAME, S3_RGB_DATABASE_PATH)
    except ClientError as e:
        logging.error(e)

    print(f"Removing temporary archive {tmp_file_name}.zip")
    os.remove(f"{tmp_file_name}.zip")


if __name__ == "__main__":
    upload_to_s3()
