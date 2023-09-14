"""The scripts in this file are needed to train the model."""
import argparse
import logging
import os

import pytorch_lightning as pl
from clearml import Task
from pytorch_lightning.callbacks import (EarlyStopping, LearningRateMonitor,
                                         ModelCheckpoint)

from src.config import Config
from src.constants import EXPERIMENTS_PATH
from src.datamodule import PosterDM
from src.lightning_module import PosterModule


def arg_parse():
    """Arg parse.

    Returns:
        _type_: _description_
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('config_file', type=str, help='config file')
    return parser.parse_args()


def train(config: Config):  # noqa:WPS210 all variables are needed
    """In this function, the model is trained.

    Args:
        config (Config): config
    """
    datamodule = PosterDM(config.data_config)
    model = PosterModule(config)

    task = Task.init(
        project_name=config.project_name,
        task_name=config.experiment_name,
        auto_connect_frameworks=True,
    )
    task.connect(config.dict())

    experiment_save_path = os.path.join(
        EXPERIMENTS_PATH,
        config.experiment_name,
    )
    os.makedirs(experiment_save_path, exist_ok=True)

    checkpoint_callback = ModelCheckpoint(
        experiment_save_path,
        monitor=config.monitor_metric,
        mode=config.monitor_mode,
        save_top_k=1,
        filename='epoch_{{epoch:02d}}-{{{monitor_metric}:.3f}}'.format(
            monitor_metric=config.monitor_metric,
        ),
    )
    trainer = pl.Trainer(
        max_epochs=config.n_epochs,
        accelerator=config.accelerator,
        devices=[config.device],
        log_every_n_steps=config.log_every_n_steps,
        callbacks=[
            checkpoint_callback,
            EarlyStopping(
                monitor=config.monitor_metric,
                patience=4,
                mode=config.monitor_mode,
            ),
            LearningRateMonitor(logging_interval='epoch'),
        ],
    )

    trainer.fit(model=model, datamodule=datamodule)
    trainer.test(
        ckpt_path=checkpoint_callback.best_model_path,
        datamodule=datamodule,
    )


if __name__ == '__main__':
    args = arg_parse()
    logging.basicConfig(level=logging.INFO)
    rand = 42
    pl.seed_everything(rand, workers=True)
    config = Config.from_yaml(args.config_file)
    train(config)
