"""This configuration is for the Pipeline."""
from typing import List

from omegaconf import OmegaConf
from pydantic import BaseModel


class LossConfig(BaseModel):
    """Config for loss.

    Args:
        BaseModel (_type_): _description_
    """
    name: str
    weight: float
    loss_fn: str
    loss_kwargs: dict


class DataConfig(BaseModel):
    """Config for data.

    Args:
        BaseModel (_type_): _description_
    """
    data_path: str
    batch_size: int
    n_workers: int
    train_size: float
    width: int
    height: int


class Config(BaseModel):
    """Config for exp.

    Args:
        BaseModel (_type_): _description_

    Returns:
        _type_: _description_
    """
    project_name: str
    experiment_name: str
    data_config: DataConfig
    n_epochs: int
    num_classes: int
    accelerator: str
    device: int
    monitor_metric: str
    monitor_mode: str
    model_kwargs: dict
    optimizer: str
    optimizer_kwargs: dict
    scheduler: str
    scheduler_kwargs: dict
    losses: List[LossConfig]

    @classmethod
    def from_yaml(cls, path: str) -> 'Config':
        """Parse data from yaml.

        Args:
            path (str): path to config

        Returns:
            Config: Config
        """
        cfg = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
        return cls(**cfg)
