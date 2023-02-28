import math
import random

from torch import Tensor
from torchvision.transforms import Normalize
from torchvision.transforms import functional as F

DEFAULT_NORM = Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
    inplace=True,
)


def square_and_rotate(src: Tensor, angle_factor: float, target_size: int) -> Tensor:
    """
    crop the tensor into square shape and rotate it

    Args:
        src (Tensor): source tensor
        target_size (int): target size. usually 224
        angle_factor (float): angle factor in [0.0,1.0]. 1.0 leads to an entire cycle.

    Returns:
        Tensor: rotated square tensor ([C,H,W]=[src,target_size,target_size])

    Note:
        `dst is src` (no copy) if there is nothing to do.
    """

    src_h, src_w = src.shape[-2:]

    # to square
    if src_h != src_w:
        if src_h > src_w:
            top = random.randint(0, src_h - src_w + 1)
            left = 0
            crop_t = src_w
        else:
            top = 0
            left = random.randint(0, src_w - src_h + 1)
            crop_t = src_h
        dst = F.crop(src, top, left, crop_t, crop_t)
    else:
        dst = src

    angle_rad = (angle_factor % 0.25) * (2 * math.pi)
    resize_t = math.ceil((math.sin(angle_rad) + math.cos(angle_rad)) * target_size)
    if not (src_h == resize_t and src_w == resize_t):
        dst = F.resize(dst, resize_t)

    if angle_factor != 0:
        angle_deg = angle_factor * 360
        dst = F.rotate(dst, angle_deg, F.InterpolationMode.BILINEAR)
        dst = F.center_crop(dst, target_size)

    return dst
