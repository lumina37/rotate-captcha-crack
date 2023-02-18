from typing import Callable

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch import Tensor
from torchvision import transforms
from torchvision.transforms import functional as F

from rotate_captcha_crack.config import CONFIG

DEFAULT_NORM = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
    inplace=True,
)

to_square: Callable[[Tensor], Tensor] = transforms.Compose(
    [
        transforms.Resize(CONFIG.dataset.img_size, antialias=True),
        transforms.RandomCrop(CONFIG.dataset.img_size),
    ]
)

img_size = CONFIG.dataset.img_size
circle_mask = Image.new('L', (img_size, img_size), color=0xFF)  # white background
circle_draw = ImageDraw.Draw(circle_mask)
circle_draw.ellipse((0, 0, img_size, img_size), fill=0)  # black circle in center

MASK = F.to_tensor(circle_mask)
MASK = MASK.to(torch.bool, copy=False)


def rotate(src: Tensor, angle: float) -> Tensor:
    dst = F.rotate(src, angle, F.InterpolationMode.BILINEAR)
    dst.masked_fill_(MASK, 0xFF)
    return dst


def rand_angles(length: int) -> Tensor:
    angle_num = CONFIG.dataset.angle_num
    unit_angle = 1 / angle_num
    angle_prob = np.full(angle_num, 1 / angle_num, dtype=np.float32)
    angles = np.random.choice(angle_num, length, p=angle_prob)
    angles = (angles * unit_angle).astype(np.float32)
    angles = torch.tensor(angles, dtype=torch.float32)
    return angles
