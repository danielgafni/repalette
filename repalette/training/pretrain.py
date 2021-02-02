import argparse
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, GPUStatsMonitor
from dotenv import load_dotenv
import os
from uuid import uuid1


from repalette.constants import (
    S3_LIGHTNING_LOGS_DIR,
    S3_MODEL_CHECKPOINTS_RELATIVE_DIR,
    LIGHTNING_LOGS_DIR,
    MODEL_CHECKPOINTS_DIR,
)
from pytorch_lightning.loggers import TensorBoardLogger
from repalette.lightning.datamodules import PreTrainDataModule
from repalette.lightning.callbacks import (
    LogPairRecoloringToTensorboard,
    Notify,
    LogHyperparamsToTensorboard,
)
from repalette.lightning.systems import PreTrainSystem
from repalette.utils.aws import upload_to_s3


def main(hparams):
    if hparams.logging_location == "s3":
        logging_dir = os.path.join(S3_LIGHTNING_LOGS_DIR, hparams.name)
    else:
        logging_dir = os.path.join(LIGHTNING_LOGS_DIR, hparams.name)

    # main LightningModule
    if hparams.checkpoint_path is not None:
        pretrain_system = PreTrainSystem.load_from_checkpoint(hparams.adversarial_system)
    else:
        pretrain_system = PreTrainSystem(**vars(hparams))

    pretrain_checkpoints = ModelCheckpoint(
        dirpath=os.path.join(MODEL_CHECKPOINTS_DIR, hparams.version),
        monitor="Val/loss",
        verbose=True,
        mode="min",
        save_top_k=hparams.save_top_k,
    )

    pretrain_early_stopping = EarlyStopping(
        monitor="Val/loss",
        min_delta=0.00,
        patience=hparams.patience,
        verbose=False,
        mode="min",
    )

    gpu_stats = GPUStatsMonitor(temperature=True)

    log_recolored_to_tensorboard = LogPairRecoloringToTensorboard()
    log_hyperparams_to_tensorboard = LogHyperparamsToTensorboard(hp_metric="Test/loss")

    notify = Notify(test_metric_name="Test/loss")

    logger = TensorBoardLogger(
        logging_dir,
        name=hparams.name,
        version=hparams.version,
        log_graph=True,
        default_hp_metric=False,
    )

    trainer = Trainer.from_argparse_args(
        hparams,
        resume_from_checkpoint=hparams.checkpoint_path,
        logger=logger,
        checkpoint_callback=pretrain_checkpoints,
        callbacks=[
            pretrain_early_stopping,
            log_recolored_to_tensorboard,
            log_hyperparams_to_tensorboard,
            gpu_stats,
            notify,
        ],
        profiler="simple",
        benchmark=True,
    )

    datamodule = PreTrainDataModule(**vars(hparams))

    trainer.fit(pretrain_system, datamodule=datamodule)

    # lightning automatically uses the best model checkpoint for testing
    trainer.test(pretrain_system, datamodule=datamodule)

    if hparams.upload_model_to_s3:
        # upload best model to S3
        best_model_path = pretrain_checkpoints.best_model_path
        S3_best_model_path = os.path.join(
            S3_MODEL_CHECKPOINTS_RELATIVE_DIR,
            hparams.name,
            ".".join([hparams.version, best_model_path.split(".")[-1]]),
        )
        upload_to_s3(best_model_path, S3_best_model_path)


if __name__ == "__main__":
    # load .env variables
    load_dotenv()

    # hyperparameters
    hparams_parser = argparse.ArgumentParser()

    # trainer
    hparams_parser.add_argument("--max-epochs", type=int, default=100)
    hparams_parser.add_argument("--gpus", type=int, default=-1)
    hparams_parser.add_argument("--precision", type=int, default=32, choices=[16, 32])
    hparams_parser.add_argument("--accumulate-grad-batches", type=int, default=1)
    hparams_parser.add_argument("--gradient-clip-val", type=float, default=0.0)
    hparams_parser.add_argument("--fast-dev-run", type=int, default=0)
    hparams_parser.add_argument("--track-grad-norm", type=int, default=-1)
    hparams_parser.add_argument("--checkpoint-path", type=str, default=None)

    # callbacks
    hparams_parser.add_argument("--patience", type=int, default=10)
    hparams_parser.add_argument("--save-top-k", type=int, default=1)

    # pretrain system
    hparams_parser = PreTrainSystem.add_argparse_args(hparams_parser)

    # datamodule
    hparams_parser = PreTrainDataModule.add_argparse_args(hparams_parser)

    # misc
    hparams_parser.add_argument(
        "--logging-location", type=str, default="s3", choices=["s3", "local"]
    )
    hparams_parser.add_argument("--upload-model-to-s3", type=bool, default=True)
    hparams_parser.add_argument("--name", type=str, default="pretrain", help="experiment name")
    hparams_parser.add_argument(
        "--version",
        type=str,
        default=None,
        help="unique! run version - used to generate checkpoint S3 path",
    )

    hparams = hparams_parser.parse_args()

    if hparams.version is None:
        hparams.version = str(uuid1())

    main(hparams)
