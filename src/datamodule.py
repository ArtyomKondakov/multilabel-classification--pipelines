import logging
import os
from typing import Optional, Tuple

import pandas as pd
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from src.augmentations import get_transforms
from src.config import DataConfig
from src.dataset import PosterDataset
from src.dataset_splitter import stratify_shuffle_split_subsets


class PosterDM(LightningDataModule):
    def __init__(self, config: DataConfig):
        super().__init__()
        self._batch_size = config.batch_size
        self._n_workers = config.n_workers
        self._train_size = config.train_size
        self._data_path = config.data_path
        self._train_transforms = get_transforms(width=config.width, height=config.height)
        self._valid_transforms = get_transforms(width=config.width, height=config.height, augmentations=False)
        self._image_folder = os.path.join(config.data_path, 'train-jpg')

        self.train_dataset: Optional[Dataset] = None
        self.valid_dataset: Optional[Dataset] = None
        self.test_dataset: Optional[Dataset] = None
        

    def prepare_data(self):
        split_and_save_datasets(self._data_path, self._train_size)

    def setup(self, stage: Optional[str] = None):
        if stage == 'fit':
            df_train = read_df(self._data_path, 'train')
            df_valid = read_df(self._data_path, 'valid')
            self.train_dataset = PosterDataset(
                df_train,
                image_folder=self._image_folder,
                transforms=self._train_transforms,
            )
            self.valid_dataset = PosterDataset(
                df_valid,
                image_folder=self._image_folder,
                transforms=self._valid_transforms,
            )

        elif stage == 'test':
            df_test = read_df(self._data_path, 'test')
            self.test_dataset = PosterDataset(
                df_test,
                image_folder=self._image_folder,
                transforms=self._valid_transforms,
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self._batch_size,
            num_workers=self._n_workers,
            shuffle=True,
            pin_memory=True,
            drop_last=False,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.valid_dataset,
            batch_size=self._batch_size,
            num_workers=self._n_workers,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self._batch_size,
            num_workers=self._n_workers,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
        )


def split_and_save_datasets(data_path: str, train_fraction: float = 0.8):
    df = pd.read_csv(os.path.join(data_path, 'train_classes.csv'))
    logging.info(f'Original dataset: {len(df)}')
    df = df.drop_duplicates()
    logging.info(f'Final dataset: {len(df)}')

    train_df, valid_df, test_df = stratify_shuffle_split_subsets(df, train_fraction=train_fraction)
    logging.info(f'Train dataset: {len(train_df)}')
    logging.info(f'Valid dataset: {len(valid_df)}')
    logging.info(f'Test dataset: {len(test_df)}')

    train_df.to_csv(os.path.join(data_path, 'df_train.csv'), index=False)
    valid_df.to_csv(os.path.join(data_path, 'df_valid.csv'), index=False)
    test_df.to_csv(os.path.join(data_path, 'df_test.csv'), index=False)
    logging.info('Datasets successfully saved!')


def read_df(data_path: str, mode: str) -> pd.DataFrame:
    return pd.read_csv(os.path.join(data_path, f'df_{mode}.csv'))
