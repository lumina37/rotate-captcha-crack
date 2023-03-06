import math
import random

import torch
from torch import Tensor
from torchvision.transforms import Normalize
from torchvision.transforms import functional as F
from torchvision.transforms import functional_tensor as F_t

from ..const import SQRT2

DEFAULT_NORM = Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
    inplace=True,
)


def to_square(src: Tensor) -> Tensor:
    """
    crop the tensor into square shape

    Args:
        src (Tensor): source tensor

    Returns:
        Tensor: square tensor ([C,H,W]=[src,shorter_edge,shorter_edge])

    Note:
        `dst is src` (no copy) if there is nothing to do.
    """

    src_h, src_w = src.shape[-2:]

    if src_h != src_w:
        if src_h > src_w:
            top = random.randint(0, src_h - src_w + 1)
            left = 0
            crop_t = src_w
        else:
            top = 0
            left = random.randint(0, src_w - src_h + 1)
            crop_t = src_h
        dst = F_t.crop(src, top, left, crop_t, crop_t)
    else:
        dst = src

    return dst


def rotate_square(src: Tensor, angle_factor: float) -> Tensor:
    """
    rotate a square tensor without any extra border

    Args:
        src (Tensor): source tensor
        angle_factor (float): angle factor in [0.0,1.0). 1.0 leads to an entire cycle.

    Returns:
        Tensor: rotated square tensor ([C,H,W]=[src,src_size/(sin(a)+cos(a)),H])

    Note:
        `dst is src` (no copy) if there is nothing to do.
    """

    src_size, src_w = src.shape[-2:]
    assert src_size == src_w

    # rotate with high resolution
    if angle_factor != 0:
        angle_deg = angle_factor * 360
        dst = F.rotate(dst, angle_deg, F.InterpolationMode.BILINEAR)
    else:
        dst = src

    # strip the border
    angle_rad = (angle_factor % 0.25) * (2 * math.pi)
    div_factor = int(math.sin(angle_rad) + math.cos(angle_rad))
    if div_factor != 1:  # int(sin(pi/2)) is identically 1
        crop_t = src_size / div_factor
        dst = F.center_crop(dst, crop_t)

    return dst


def crop_rotate_resize(src: Tensor, angle_factor: float, target_size: int = 224) -> Tensor:
    """
    crop the tensor into square shape
    then rotate it without any extra border
    then resize it to the target size

    Args:
        src (Tensor): source tensor ([C,H,W]=[ud,ud,ud], dtype=uint8, range=[0,255])
        angle_factor (float): angle factor in [0.0,1.0). 1.0 leads to an entire cycle.
        target_size (int, optional): target size. Defaults to 224.

    Returns:
        Tensor: tensor ([C,H,W]=[src,target_size,target_size], dtype=float32, range=[0.0,1.0))
    """

    img_ts = to_square(src)
    img_ts.to(dtype=torch.float32).div_(255)
    img_ts = rotate_square(img_ts, angle_factor, target_size)
    img_ts = F_t.resize(img_ts, [target_size, target_size], antialias=True)


def strip_circle_border(src: Tensor, target_size: int = 224) -> Tensor:
    """
    strip the circle border

    Args:
        src (Tensor): source tensor
        target_size (int, optional): target size. Defaults to 224.

    Returns:
        Tensor: striped tensor ([C,H,W]=[src,target_size,target_size])
    """

    src_h, src_w = src.shape[-2:]
    assert src_h == src_w

    src_size = src_h
    dst = F.center_crop(src, src_size / SQRT2)
    dst = F.resize(dst, target_size)

    return dst
