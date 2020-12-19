from cosmos.api import Cosmos
import argparse
from dotenv import load_dotenv
import os
from uuid import uuid1
import sys

from repalette.constants import RDS_COSMOS_DATABASE, S3_BUCKET_PATH

"""
This script can be modified to send different jobs to AWS Batch via Cosmos.
"""


env_variables = [
            "AWS_DEFAULT_REGION",
            "S3_BUCKET_NAME",
        ]


def pretrain(version, num_workers, batch_size, multiplier, size, max_epochs=None):
    """
    Main function for Cosmos to execute
    """
    if max_epochs:
        max_epochs_part = f"--max-epochs {max_epochs}"
    else:
        max_epochs_part = ""

    command = f"""
    df -h
    poetry run python repalette/db/utils/download_rgb_from_s3.py
    poetry run python scripts/pretrain.py --version {version} --num-workers {num_workers} {max_epochs_part} --batch-size {batch_size} --multiplier {multiplier} --size {size}
    """
    return command


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--name", type=str, default="pretrain", help="Cosmos workflow name"
    )
    parser.add_argument(
        "--gpu-req",
        type=int,
        default=1,
        help="Number of GPUs",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=20,
        help="Number of attempts to run the tasks",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=1,
        help="Number of epochs to train",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--multiplier",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--size",
        type=float,
        default=1.,
    )
    parser.add_argument("--core-req", type=int, default=8)
    parser.add_argument("--mem-req", type=int, default=32000)
    args = parser.parse_args()

    load_dotenv()

    def set_env_variables():

        return "\n".join(
            [f'export {variable}="{os.getenv(variable)}"' for variable in env_variables]
        )

    cosmos = Cosmos(
        RDS_COSMOS_DATABASE,
        default_drm="awsbatch",
        default_drm_options=dict(
            container_image=os.getenv("ECR_CONTAINER_IMAGE"),
            s3_prefix_for_command_script_temp_files=os.path.join(
                S3_BUCKET_PATH, "cosmos-tmp"
            ),
            shm_size=int(args.mem_req * 0.75),
            retry_only_if_status_reason_matches="Host EC2 .+ terminated." # only retry on spot instance death
        ),
        default_queue=os.getenv("BATCH_QUEUE_NAME")
    )
    cosmos.initdb()

    workflow_name = f"pretrain-{uuid1().hex}"
    workflow = cosmos.start(workflow_name, restart=True, skip_confirm=True)

    task_name = uuid1().hex

    workflow.add_task(
        func=pretrain,
        params=dict(
            version=task_name,
            max_epochs=args.max_epochs,
            num_workers=args.core_req - 1,
            batch_size=args.batch_size,
            multiplier=args.multiplier,
            size=args.size
        ),
        uid=task_name,
        time_req=None,
        max_attempts=args.max_attempts,
        core_req=args.core_req,
        gpu_req=args.gpu_req,
        mem_req=args.mem_req,
        environment_variables={var: os.getenv(var) for var in env_variables}
    )

    workflow.run()

    sys.exit(0 if workflow.successful else 1)
