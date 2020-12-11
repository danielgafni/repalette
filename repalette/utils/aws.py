import boto3
from botocore.exceptions import ClientError
import logging

from repalette.constants import S3_BUCKET_NAME


def upload_to_s3(file_path, bucket_path, bucket_name=S3_BUCKET_NAME):
    s3_client = boto3.client("s3")
    try:
        print(f"Uploading {file_path} to s3://{bucket_name}/{bucket_path}")
        s3_client.upload_file(file_path, bucket_name, bucket_path)
    except ClientError as e:
        logging.error(e)
