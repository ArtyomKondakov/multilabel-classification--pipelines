"""This script is needed to make augmentations of images."""
from typing import Union

import albumentations as albu
from albumentations.pytorch import ToTensorV2

TRANSFORM_TYPE = Union[albu.BasicTransform, albu.BaseCompose]


def get_transforms(
    width: int,
    height: int,
    preprocessing: bool = True,
    augmentations: bool = True,
    postprocessing: bool = True,
) -> TRANSFORM_TYPE:
    """Return the transforms images.

    Args:
        width (int): width image
        height (int): height image
        preprocessing (bool): preprocessing - resize image.
        augmentations (bool): augmentations for image.
        postprocessing (bool): Normalize and ToTensorV2.

    Returns:
        TRANSFORM_TYPE: _description_
    """
    transforms = []
    if preprocessing:
        transforms.append(albu.Resize(height=height, width=width))

    if augmentations:
        transforms.extend(
            [
                albu.HorizontalFlip(p=0.5),
                albu.VerticalFlip(p=0.5),
                albu.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),  # noqa: WPS432, E501
                albu.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),  # noqa: WPS432, E501
                albu.ShiftScaleRotate(),
                albu.GaussianBlur(),
            ],
        )

    if postprocessing:
        transforms.extend([albu.Normalize(), ToTensorV2()])

    return albu.Compose(transforms)
