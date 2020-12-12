from cosmos.api import Cosmos
import argparse
from dotenv import load_dotenv
import os
from uuid import uuid1
import sys

from repalette.constants import DEFAULT_COSMOS_DATABASE, S3_BUCKET_PATH


def pretrain(version, num_workers=7, max_epochs=None):
    if max_epochs:
        max_epochs_part = f"--max-epochs {max_epochs}"
    else:
        max_epochs_part = ""

    command = f"""
    {set_env_variables()}
    poetry run python repalette/db/utils/download_rgb_from_s3.py
    poetry run python scripts/pretrain.py --version {version} --num-workers {num_workers} {max_epochs_part}
    """
    return command


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--name", type=str, default="pretrain", help="Cosmos workflow name"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=7,
        help="Number of workers for the dataloaders",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=20,
        help="Number of attempts to run the tasks",
    )
    parser.add_argument("--core-req", type=int, default=8)
    parser.add_argument("--mem-req", type=int, default=16000)
    args = parser.parse_args()

    load_dotenv()

    def set_env_variables():
        env_variables = [
            "AWS_DEFAULT_REGION",
            "S3_BUCKET_NAME",
        ]
        return "\n".join(
            [f'export {variable}="{os.getenv(variable)}"' for variable in env_variables]
        )

    cosmos = Cosmos(
        DEFAULT_COSMOS_DATABASE,
        default_drm="awsbatch",
        default_drm_options=dict(
            container_image=os.getenv("ECR_CONTAINER_IMAGE"),
            s3_prefix_for_command_script_temp_files=os.path.join(
                S3_BUCKET_PATH, "cosmos-tmp"
            ),
            # only retry on spot instance death
            retry_only_if_status_reason_matches="Host EC2 .+ terminated.",
        ),
        default_queue=os.getenv("BATCH_QUEUE_NAME"),
    )
    cosmos.initdb()

    workflow = cosmos.start(args.name, restart=True, skip_confirm=True)

    task_name = uuid1().hex

    workflow.add_task(
        func=pretrain,
        params=dict(
            version=task_name,
            max_epochs=1,
            num_workers=args.num_workers,
        ),
        uid=task_name,
        time_req=None,
        max_attempts=args.max_attempts,
        core_req=args.core_req,
        mem_req=args.mem_req,
    )

    workflow.run()

    sys.exit(0 if workflow.successful else 1)
