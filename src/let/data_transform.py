from monai.transforms import (
    EnsureChannelFirstd,
    Compose,
    EnsureTyped,
    RandRotated,
    RandFlipd
)
import numpy as np


def get_preprocess_transforms():
    transforms = [
        # make sure we have all tensors, no longer numpy arrays
        EnsureTyped(keys=["ct", "dose", "let"], data_type="tensor",
                    allow_missing_keys=True),

        # add channel first
        EnsureChannelFirstd(keys=["ct", "dose", "let"],
                            channel_dim="no_channel",
                            allow_missing_keys=True),
    ]

    return Compose(transforms)


def get_augmentation_transforms(p=0.5):
    # NOTE: we would have to rotate the CT and the Dose and LET
    # in the same way! Similarly with Axis flipping.
    # This seems to be covered by MONAI internally
    # with the dictionary transforms

    transforms = [
        # only rotate along the first dimension
        # (which is actually z, but denoted as x in monai)
        RandRotated(
            keys=["ct", "dose", "let"], prob=p,
            range_x=(np.deg2rad(-30), np.deg2rad(30)),
            range_y=0,
            range_z=0,
            allow_missing_keys=True
        ),
        # flip front - back
        RandFlipd(
            keys=["ct", "dose", "let"],
            prob=p,
            spatial_axis=[1],
            allow_missing_keys=True
        ),
        # left - right
        RandFlipd(
            keys=["ct", "dose", "let"],
            prob=p,
            spatial_axis=[2],
            allow_missing_keys=True
        )
    ]

    # TODO: further transforms from
    # https://docs.monai.io/en/stable/transforms.html#module-monai.transforms
    # might be suitable, but did not choose them for the sake of not
    # introducing any wrong information into the images

    return Compose(transforms)
