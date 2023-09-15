"""
Donut
Copyright (c) 2022-present NAVER Corp.
MIT License
Copyright (c) Meta Platforms, Inc. and affiliates.
"""
import argparse
import datetime
import os
from os.path import basename
from pathlib import Path

import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    Callback,
    GradientAccumulationScheduler,
)
from lightning.pytorch.loggers.tensorboard import TensorBoardLogger
from lightning.pytorch.plugins import CheckpointIO
from lightning.pytorch.plugins.environments import SLURMEnvironment
from lightning.pytorch.utilities import rank_zero_only
from sconf import Config

from nougat import NougatDataset
from lightning_module import NougatDataPLModule, NougatModelPLModule

try:
    import wandb
    from lightning.pytorch.loggers import WandbLogger as Logger
except ModuleNotFoundError:
    from lightning.pytorch.loggers.tensorboard import TensorBoardLogger as Logger

import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)


class CustomCheckpointIO(CheckpointIO):
    """
    A custom class for saving and loading checkpoints with additional functionality.

    Args:
        `CheckpointIO` (class): The base class for checkpoint I/O operations.

    Methods:
        `save_checkpoint(checkpoint, path, storage_options=None)`:
            Save a checkpoint to the specified path.

        `load_checkpoint(path, storage_options=None)`:
            Load a checkpoint from the specified path.

        `remove_checkpoint(path) -> None`:
            Remove a checkpoint from the specified path.

    """

    @rank_zero_only
    def save_checkpoint(self, checkpoint, path, storage_options=None):
        """
        Save a checkpoint to the specified path.

        Args:
            `checkpoint` (dict): The dictionary containing the checkpoint data.
            `path` (str): The path where the checkpoint will be saved.
            `storage_options` (dict, optional): Additional storage options.
        """
        torch.save(checkpoint, path)

    def load_checkpoint(self, path, storage_options=None):
        """
        Load a checkpoint from the specified path.

        Args:
            `path` (str): The path from which the checkpoint will be loaded.
            `storage_options` (dict, optional): Additional storage options.
        """
        path = Path(path)
        if path.is_file():
            print("path:", path, path.is_dir())
            ckpt = torch.load(path)
            if not "state_dict" in ckpt:
                ckpt["state_dict"] = {
                    "model." + key: value
                    for key, value in torch.load(
                        path.parent / "pytorch_model.bin"
                    ).items()
                }
            return ckpt
        else:
            checkpoint = torch.load(path / "artifacts.ckpt")
            state_dict = torch.load(path / "pytorch_model.bin")
            checkpoint["state_dict"] = {
                "model." + key: value for key, value in state_dict.items()
            }
            return checkpoint

    def remove_checkpoint(self, path) -> None:
        return super().remove_checkpoint(path)


class GradNormCallback(Callback):
    """
    Logs the gradient norm.
    """

    @staticmethod
    def gradient_norm(model):
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.detach().data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm**0.5
        return total_norm

    def on_after_backward(self, trainer, model):
        model.log("train/grad_norm", self.gradient_norm(model))


@rank_zero_only
def save_config_file(config, path):
    if not Path(path).exists():
        os.makedirs(path)
    save_path = Path(path) / "config.yaml"
    print(config.dumps())
    with open(save_path, "w") as f:
        f.write(config.dumps(modified_color=None, quote_str=True))
        print(f"Config is saved at {save_path}")


def train(config):
    """
    Train a Nougat model using the provided configuration.

    Args:
        `config` (dict): A dictionary containing configuration settings for training.
    """
    pl.seed_everything(config.get("seed", 42), workers=True)

    model_module = NougatModelPLModule(config)
    data_module = NougatDataPLModule(config)

    # add datasets to data_module
    datasets = {"train": [], "validation": []}
    for i, dataset_path in enumerate(config.dataset_paths):
        for split in ["train", "validation"]:
            datasets[split].append(
                NougatDataset(
                    dataset_path=dataset_path,
                    nougat_model=model_module.model,
                    max_length=config.max_length,
                    split=split,
                )
            )
    data_module.train_datasets = datasets["train"]
    data_module.val_datasets = datasets["validation"]

    lr_callback = LearningRateMonitor(logging_interval="step")

    checkpoint_callback = ModelCheckpoint(
        save_last=True,
        dirpath=Path(config.result_path) / config.exp_name / config.exp_version,
    )
    grad_norm_callback = GradNormCallback()
    custom_ckpt = CustomCheckpointIO()

    if not config.debug:
        logger = Logger(config.exp_name, project="Nougat", config=dict(config))
    else:
        logger = TensorBoardLogger(
            save_dir=config.result_path,
            name=config.exp_name,
            version=config.exp_version,
            default_hp_metric=False,
        )
    trainer = pl.Trainer(
        num_nodes=config.get("num_nodes", 1),
        devices="auto",
        strategy="ddp_find_unused_parameters_true",
        accelerator="auto",
        # plugins=[SLURMEnvironment(auto_requeue=False)],
        max_epochs=config.max_epochs,
        max_steps=config.max_steps,
        val_check_interval=config.val_check_interval,
        check_val_every_n_epoch=config.check_val_every_n_epoch,
        limit_val_batches=config.val_batches,
        gradient_clip_val=config.gradient_clip_val,
        log_every_n_steps=15,
        precision="bf16-mixed",
        num_sanity_val_steps=0,
        logger=logger,
        callbacks=[
            lr_callback,
            grad_norm_callback,
            checkpoint_callback,
            GradientAccumulationScheduler({0: config.accumulate_grad_batches}),
        ],
    )

    trainer.fit(
        model_module,
        data_module,
        ckpt_path=config.get("resume_from_checkpoint_path", None),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--exp_version", type=str, required=False)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--job", type=int, default=None)
    args, left_argv = parser.parse_known_args()

    config = Config(args.config)
    config.argv_update(left_argv)
    config.debug = args.debug
    config.job = args.job
    if not config.get("exp_name", False):
        config.exp_name = basename(args.config).split(".")[0]
    config.exp_version = (
        datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        if not args.exp_version
        else args.exp_version
    )

    save_config_file(
        config, Path(config.result_path) / config.exp_name / config.exp_version
    )
    train(config)
