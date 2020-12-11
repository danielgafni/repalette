import argparse
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, GPUStatsMonitor
from dotenv import load_dotenv
import os

from repalette.constants import (
    DEFAULT_PRETRAIN_LR,
    DEFAULT_PRETRAIN_BETA_1,
    DEFAULT_PRETRAIN_BETA_2,
    S3_LIGHTNING_LOGS_DIR,
    S3_MODEL_CHECKPOINTS_RELATIVE_DIR,
    DEFAULT_PRETRAIN_WEIGHT_DECAY,
    MODEL_CHECKPOINTS_DIR,
)
from pytorch_lightning.loggers import TensorBoardLogger
from repalette.lightning.datamodules import PreTrainDataModule
from repalette.lightning.callbacks import LogRecoloringToTensorboard
from repalette.lightning.systems import PreTrainSystem
from repalette.utils.aws import upload_to_s3


if __name__ == "__main__":
    # load .env variables
    load_dotenv()

    # hyperparameters
    hparams_parser = argparse.ArgumentParser()

    # trainer
    hparams_parser.add_argument("--max-epochs", type=int, default=None)
    hparams_parser.add_argument("--gpus", type=int, default=-1)
    hparams_parser.add_argument("--precision", type=int, default=16, choices=[16, 32])
    hparams_parser.add_argument("--accumulate-grad-batches", type=int, default=1)
    hparams_parser.add_argument("--gradient-clip-val", type=float, default=0.0)
    # hparams_parser.add_argument(
    #     "--auto-lr-find", type=str, default=False, choices=["learning_rate", False]
    # )
    # hparams_parser.add_argument(
    #     "--auto-scale-batch-size",
    #     type=str,
    #     default=None,
    #     choices=[None, "power", "binsearch"],
    # )

    # callbacks
    hparams_parser.add_argument("--patience", type=int, default=20)
    hparams_parser.add_argument("--save-top-k", type=int, default=1)

    # pretrain task
    hparams_parser.add_argument(
        "--learning-rate", type=float, default=DEFAULT_PRETRAIN_LR
    )
    hparams_parser.add_argument("--beta-1", type=float, default=DEFAULT_PRETRAIN_BETA_1)
    hparams_parser.add_argument("--beta-2", type=float, default=DEFAULT_PRETRAIN_BETA_2)
    hparams_parser.add_argument(
        "--weight-decay", type=float, default=DEFAULT_PRETRAIN_WEIGHT_DECAY
    )
    hparams_parser.add_argument(
        "--optimizer", type=str, default="adam", choices=["adam", "adamw"]
    )
    hparams_parser.add_argument("--scheduler-patience", type=int, default=10)
    hparams_parser.add_argument("--batch-size", type=int, default=8)
    hparams_parser.add_argument("--multiplier", type=int, default=16)

    # datamodule
    hparams_parser.add_argument("--num-workers", type=int, default=7)
    hparams_parser.add_argument("--shuffle", type=bool, default=True)
    hparams_parser.add_argument("--size", type=float, default=1.0)
    hparams_parser.add_argument("--pin-memory", type=bool, default=True)

    # misc
    hparams_parser.add_argument(
        "--name", type=str, default="pretrain", help="experiment name"
    )
    hparams_parser.add_argument("--version", type=str, default="0", help="run version - used to generate checkpoint S3 path")
    hparams_parser.add_argument(
        "--logger", type=str, default="tensorboard", choices=["tensorboard"]
    )

    hparams = hparams_parser.parse_args()

    # main LightningModule
    pretrain_system = PreTrainSystem(
        learning_rate=hparams.learning_rate,
        beta_1=hparams.beta_1,
        beta_2=hparams.beta_2,
        weight_decay=hparams.weight_decay,
        optimizer=hparams.optimizer,
        batch_size=hparams.batch_size,
        multiplier=hparams.multiplier,
        scheduler_patience=hparams.scheduler_patience,
    )

    pretrain_checkpoints = ModelCheckpoint(
        dirpath=MODEL_CHECKPOINTS_DIR,
        monitor="Val/loss_epoch",
        verbose=True,
        mode="min",
        save_top_k=hparams.save_top_k,
    )

    pretrain_early_stopping = EarlyStopping(
        monitor="Val/loss_epoch",
        min_delta=0.00,
        patience=hparams.patience,
        verbose=False,
        mode="min",

    )

    pretrain_gpu_stats_monitor = GPUStatsMonitor(temperature=True)

    log_recoloring_to_tensorboard = LogRecoloringToTensorboard()

    logger = TensorBoardLogger(
        S3_LIGHTNING_LOGS_DIR,
        name=hparams.name,
        version=hparams.version,
        log_graph=True,
        default_hp_metric=False
    )

    trainer = Trainer.from_argparse_args(
        hparams,
        logger=logger,
        checkpoint_callback=pretrain_checkpoints,
        callbacks=[
            pretrain_early_stopping,
            log_recoloring_to_tensorboard,
            pretrain_gpu_stats_monitor,
        ],
        profiler="simple"
    )

    datamodule = PreTrainDataModule(
        batch_size=pretrain_system.hparams.batch_size,
        multiplier=pretrain_system.hparams.multiplier,
        shuffle=hparams.shuffle,
        num_workers=hparams.num_workers,
        size=hparams.size,
        pin_memory=hparams.pin_memory,
    )

    # trainer.tune(pretrain_system, datamodule=datamodule)

    trainer.fit(pretrain_system, datamodule=datamodule)

    # get best checkpoint
    best_model_path = pretrain_checkpoints.best_model_path

    pretrain_system = PreTrainSystem.load_from_checkpoint(best_model_path)

    test_result = trainer.test(pretrain_system, datamodule=datamodule)
    pretrain_system.hparams.hp_metric = test_result[0]["Test/loss_epoch"]
    logger.log_hyperparams(pretrain_system.hparams)
    logger.finalize(status="success")

    # upload best model to S3
    S3_best_model_path = os.path.join(
        S3_MODEL_CHECKPOINTS_RELATIVE_DIR,
        hparams.name,
        ".".join([hparams.version, best_model_path.split(".")[-1]]),
    )
    upload_to_s3(best_model_path, S3_best_model_path)
