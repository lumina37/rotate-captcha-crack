import dataclasses as dcs
import math
from collections.abc import Iterator
from typing import Generic, TypeVar, Union

import torch
from torch import Tensor

from ...const import DEFAULT_CLS_NUM

TLabel = TypeVar('TLabel')


@dcs.dataclass
class ImgWithLabel(Generic[TLabel]):
    """
    Container of an image tensor and its label.

    Args:
        img (Tensor): image tensor
        label (TLabel): related label

    Note:
        Can be converted to `tuple` via the iterator protocol.

    Example:
        `tuple(ImgWithLabel(img, 0.25)) == (img, 0.25)`
    """

    img: Tensor
    label: TLabel

    def __iter__(self) -> Iterator[Union[Tensor, TLabel]]:
        return iter((self.img, self.label))


@dcs.dataclass
class ScalarLabel:
    """
    Simply convert the scalar into a tensor, which is commonly used for L1/MSELoss.

    Methods:
        - `self(data: ImgWithLabel[float]) -> ImgWithLabel[Tensor]` \\
            `ret.img` is the same as `data.img`, this function won't modify the image tensor.
            `data.label` is the angle factor (float, range=[0.0,1.0)), where 1.0 means an entire cycle.
            `ret.label` is the angle factor in tensor ([C]=[1], dtype=float32, range=[0.0,1.0)).
    """

    def __call__(self, data: ImgWithLabel[float]) -> ImgWithLabel[Tensor]:
        label_ts = torch.tensor(data.label)
        return ImgWithLabel(data.img, label_ts)


@dcs.dataclass
class OnehotLabel:
    """
    Convert the scalar `angle_factor` label to one-hot label, which is commonly used for CrossEntropyLoss.

    Args:
        cls_num (int, optional): divide into how many classes. Default to `DEFAULT_CLS_NUM`.

    Methods:
        - `self(data: ImgWithLabel[float]) -> ImgWithLabel[Tensor]` \\
            `ret.img` is the same as `data.img`, this function won't modify the image tensor.
            `data.label` is the angle factor (float, range=[0.0,1.0)), where 1.0 means an entire cycle.
            `ret.label` is the one-hot label ([C]=[1], dtype=float32, range=[0.0,cls_num)).
    """

    cls_num: int = DEFAULT_CLS_NUM

    def __call__(self, data: ImgWithLabel[float]) -> ImgWithLabel[Tensor]:
        label_idx = data.label * self.cls_num
        label_ts = torch.tensor(label_idx)
        return ImgWithLabel(data.img, label_ts)


@dcs.dataclass
class CircularSmoothLabel:
    """
    Convert the scalar `angle_factor` label to CSL (Circular Smooth Label), which is an optimized label for CrossEntropyLoss.

    Args:
        cls_num (int, optional): divide into how many classes. Defaults to `DEFAULT_CLS_NUM`.
        std (float, optional): standard deviation of the normal distribution after smoothing. Defaults to 0.5.

    Methods:
        - `self(data: ImgWithLabel[float]) -> ImgWithLabel[Tensor]` \\
            `ret.img` is the same as `data.img`, this function won't modify the image tensor.
            `data.label` is the angle factor (float, range=[0.0,1.0)), where 1.0 means an entire cycle.
            `ret.label` is the CSL ([C]=[cls_num], dtype=float32, range=[0.0,1.0)).

    Reference:
        CrossEntropyLoss ensures a uniform distance between each labels, e.g. $dist(1°,180°) \\eq dist(1°,3°)$, which solves the circular issue. \\
        Based on that, *[Arbitrary-Oriented Object Detection with Circular Smooth Label (ECCV'20)](https://www.researchgate.net/profile/Xue-Yang-69/publication/343636147_Arbitrary-Oriented_Object_Detection_with_Circular_Smooth_Label/links/5f46456b458515b7295797fd/Arbitrary-Oriented-Object-Detection-with-Circular-Smooth-Label.pdf)* introduces a further improvement, by smoothing the one-hot label, e.g. `[0,1,0,0] -> [0.1,0.8,0.1,0]`, CSL provides a loss measurement closer to our intuition, \\
        such that $\\mathrm{dist}(1°,180°) \\gt \\mathrm{dist}(1°,3°)$.
    """

    cls_num: int = DEFAULT_CLS_NUM
    std: float = 1.0
    normal_dist: Tensor = dcs.field(default=None, init=False)

    def __post_init__(self) -> None:
        x = torch.linspace(0, self.cls_num - 1, self.cls_num)
        dividend = math.sqrt(2.0 * torch.pi * self.std**2)
        self.normal_dist = torch.exp(-((x - self.cls_num / 2) ** 2) / (2 * self.std**2)) / dividend

    def __call__(self, data: ImgWithLabel[float]) -> ImgWithLabel[Tensor]:
        label_idx = data.label * self.cls_num

        # rolling right shift
        shift = label_idx - self.cls_num / 2
        label_ts = self.normal_dist.clone()
        label_ts = torch.roll(label_ts, shifts=round(shift), dims=0)

        return ImgWithLabel(data.img, label_ts)
