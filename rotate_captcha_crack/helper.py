import math
import random

import numpy as np
import torch
from torch import Tensor
from torchvision import transforms
from torchvision.transforms import functional as F

DEFAULT_NORM = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
    inplace=True,
)


def square_and_rotate(src: Tensor, target_size: int, angle_factor: float) -> Tensor:
    """
    crop the tensor into square shape and rotate it

    Args:
        src (Tensor): source tensor
        target_size (int): target size. usually 224
        angle_factor (float): angle factor in [0.0,1.0]

    Returns:
        Tensor: rotated square tensor

    Note:
        `dst is src` if there is nothing to do, without copy.
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


def generate_angles(ori_size: int, angle_num: int, copy_num: int) -> Tensor:
    dst_size = ori_size * copy_num
    dst_array = np.empty(dst_size, dtype=np.float32)
    rng = np.random.default_rng()

    for i in range(0, dst_size, copy_num):
        group = rng.choice(angle_num, copy_num, replace=False)
        group = group.astype(np.float32)
        np.divide(group, angle_num, group)
        dst_array[i : i + copy_num] = group

    dst = torch.from_numpy(dst_array)

    return dst
