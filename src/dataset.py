"""Dataset class for pipeline."""
import os
from typing import Optional, Union

import albumentations as albu
import cv2
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

TRANSFORM_TYPE = Union[albu.BasicTransform, albu.BaseCompose]


class PosterDataset(Dataset):
    """Dataset class for pipeline.

    Args:
        Dataset (_type_): pytorch dataset class
    """
    def __init__(
        self,
        df: pd.DataFrame,
        image_folder: str,
        transforms: Optional[TRANSFORM_TYPE] = None,
    ):
        """Initialize an instance of the class.

        Args:
            df (pd.DataFrame): data frmae
            image_folder (str): path to image folder
            transforms (Optional[TRANSFORM_TYPE], optional): augmentation
        """
        self.df = df
        self.image_folder = image_folder
        self.transforms = transforms

    def __getitem__(self, idx: int):
        """For iterat a dataset.

        Args:
            idx (int): index - the number of the dataframe line

        Returns:
            _type_: image and labels
        """
        row = self.df.iloc[idx]

        image_path = os.path.join(
            self.image_folder,
            '{image_name}.jpg'.format(image_name=row.image_name),
        )
        labels = np.array(row.values[1:], dtype='float32')
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        data_im = {'image': image, 'labels': labels}

        if self.transforms:
            data_im = self.transforms(**data_im)

        return data_im['image'], data_im['labels']

    def __len__(self) -> int:
        """Show dataset size.

        Returns:
            int: idx last img
        """
        return len(self.df)
