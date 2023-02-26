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

    angle_rad = angle_factor * math.pi
    resize_t = math.ceil((abs(math.sin(angle_rad)) + abs(math.cos(angle_rad))) * target_size)
    if not (src_h == resize_t and src_w == resize_t):
        dst = F.resize(dst, target_size)

    if angle_factor != 0:
        angle_deg = angle_factor * 360
        dst = F.rotate(dst, angle_deg, F.InterpolationMode.BILINEAR)
        dst = F.center_crop(dst, target_size)

    return dst


def rand_angles(length: int) -> Tensor:
    angle_num = 4
    unit_angle = 1 / angle_num
    angle_prob = np.full(angle_num, 1 / angle_num, dtype=np.float32)
    angles = np.random.choice(angle_num, length, p=angle_prob)
    angles = (angles * unit_angle).astype(np.float32)
    angles = torch.tensor(angles, dtype=torch.float32)
    return angles
