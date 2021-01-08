import boto3
from botocore.exceptions import ClientError
import logging

from repalette.constants import S3_BUCKET_NAME


INCREASE_DOCKER_SHARED_MEMORY = """
SYSTEM_MEMORY_MB_75_PERCENT=$(free|head -2|tail -1|awk '{ print $2*.75M }')
mount -o remount,size=$SYSTEM_MEMORY_MB_75_PERCENT /dev/shm
"""


def upload_to_s3(
    file_path,
    bucket_path,
    bucket_name=S3_BUCKET_NAME,
):
    s3_client = boto3.client("s3")
    try:
        print(f"Uploading {file_path} to s3://{bucket_name}/{bucket_path}")
        s3_client.upload_file(file_path, bucket_name, bucket_path)
    except ClientError as e:
        logging.error(e)
