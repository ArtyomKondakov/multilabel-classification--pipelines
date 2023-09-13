"""Datamodule for the pipeline."""
import logging
import os
from typing import Optional

import pandas as pd
from pytorch_lightning import LightningDataModule
from sklearn.preprocessing import MultiLabelBinarizer
from torch.utils.data import DataLoader, Dataset

from src.augmentations import get_transforms
from src.config import DataConfig
from src.dataset import PosterDataset
from src.dataset_splitter import stratify_shuffle_split_subsets


class PosterDM(LightningDataModule):
    """Сlass for processing with data.

    Args:
        LightningDataModule (_type_): The LightningDataModule is a convenient
        way to manage data
    """
    def __init__(self, config: DataConfig):
        """Initialize an instance of the class.

        Args:
            config (DataConfig): config data
        """
        super().__init__()
        self._batch_size = config.batch_size
        self._n_workers = config.n_workers
        self._train_size = config.train_size
        self._data_path = config.data_path
        self._train_transforms = get_transforms(
            width=config.width, height=config.height,
        )
        self._valid_transforms = get_transforms(
            width=config.width, height=config.height, augmentations=False,
        )
        self._image_folder = os.path.join(config.data_path, 'train-jpg')

        self.train_dataset: Optional[Dataset] = None
        self.valid_dataset: Optional[Dataset] = None
        self.test_dataset: Optional[Dataset] = None

    def prepare_data(self):
        """For split and save datasets."""
        split_and_save_datasets(self._data_path, self._train_size)

    def setup(self, stage: Optional[str] = None):
        """Create a dataset class.

        Args:
            stage (Optional[str], optional):
            Shows the training or test is in progress. Defaults to None.
        """
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
        """Train dataloader.

        Returns:
            DataLoader: The most important argument of Data Loader constructor
            is dataset, which indicates a dataset object to load data from.
        """
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self._batch_size,
            num_workers=self._n_workers,
            shuffle=True,
            pin_memory=True,
            drop_last=False,
        )

    def val_dataloader(self) -> DataLoader:
        """Val dataloader.

        Returns:
            DataLoader: The most important argument of Data Loader constructor
            is dataset, which indicates a dataset object to load data from.
        """
        return DataLoader(
            dataset=self.valid_dataset,
            batch_size=self._batch_size,
            num_workers=self._n_workers,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
        )

    def test_dataloader(self) -> DataLoader:
        """Test dataloader.

        Returns:
            DataLoader: The most important argument of Data Loader constructor
            is dataset, which indicates a dataset object to load data from.
        """
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self._batch_size,
            num_workers=self._n_workers,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
        )


def one_hot_encoding(df_train):
    """With this function, you can make one hot encoding.

    Args:
        df_train (_type_): dataset

    Returns:
        _type_: pd.Dataframe
    """
    df_train['list_tags'] = df_train.tags.str.split(' ')
    encoder = MultiLabelBinarizer()
    ohe_tags_train = encoder.fit_transform(df_train.list_tags.values)
    df_train['ohe_tags'] = ohe_tags_train.tolist()
    df_onehot = pd.DataFrame(
        encoder.transform(df_train['list_tags']), columns=encoder.classes_,
    )
    df = df_train.join(df_onehot)
    return df.drop(['list_tags', 'ohe_tags'], axis=1)


def save_datasets(train_df, valid_df, test_df, data_path):
    """With this function, you can save the datasets.

    Args:
        train_df (_type_): train dataset
        valid_df (_type_): valid dataset
        test_df (_type_): test dataset
        data_path (_type_): path to data
    """
    train_df.to_csv(os.path.join(data_path, 'df_train.csv'), index=False)
    valid_df.to_csv(os.path.join(data_path, 'df_valid.csv'), index=False)
    test_df.to_csv(os.path.join(data_path, 'df_test.csv'), index=False)


def split_and_save_datasets(data_path: str, train_fraction: float = 0.8):
    """With this function, you can split the dataset into selections.

    Args:
        data_path (str): path to data
        train_fraction (float): size for the training sample.
    """
    df = pd.read_csv(os.path.join(data_path, 'train_classes.csv'))
    df = one_hot_encoding(df)
    logging.info('Original dataset: {len_df}'.format(len_df=len(df)))
    df = df.drop_duplicates()
    logging.info('Final dataset: {len_df}'.format(len_df=len(df)))

    train_df, valid_df, test_df = stratify_shuffle_split_subsets(
        df, train_fraction=train_fraction,
    )
    logging.info('Train dataset: {len_df}'.format(len_df=len(train_df)))
    logging.info('Valid dataset: {len_df}'.format(len_df=len(valid_df)))
    logging.info('Test dataset: {len_df}'.format(len_df=len(test_df)))

    save_datasets(train_df, valid_df, test_df, data_path)
    logging.info('Datasets successfully saved!')


def read_df(data_path: str, mode: str) -> pd.DataFrame:
    """With this function, you can read dataset.

    Args:
        data_path (str): path to data
        mode (str): train, test or val

    Returns:
        pd.DataFrame: dataframe
    """
    df_format = 'df_{mode}.csv'.format(mode=mode)
    path_to_df = os.path.join(data_path, df_format)
    return pd.read_csv(path_to_df)
