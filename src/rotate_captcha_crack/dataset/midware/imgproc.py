import math
import random

from torch import Tensor
from torchvision.transforms import Normalize
from torchvision.transforms import functional as F
from torchvision.transforms.functional import F_t

from ...const import DEFAULT_TARGET_SIZE, SQRT2
from .totensor import u8_to_float32

DEFAULT_NORM = Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
    inplace=True,
)


def to_square(src: Tensor) -> Tensor:
    """
    Crop the tensor into square shape.

    Args:
        src (Tensor): tensor

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


def rotate_by_factor(src: Tensor, angle_factor: float) -> Tensor:
    """
    Rotate a square tensor.

    Args:
        src (Tensor): square tensor
        angle_factor (float): angle factor in [0.0,1.0). 1.0 means an entire cycle.

    Returns:
        Tensor: rotated tensor ([C,H,W]=[src,src,src])

    Note:
        `dst is src` (no copy) if there is nothing to do.
    """

    if angle_factor != 0:
        angle_deg = angle_factor * 360
        dst = F.rotate(src, angle_deg, F.InterpolationMode.BILINEAR)
    else:
        dst = src

    return dst


def rotate_square(src: Tensor, angle_factor: float) -> Tensor:
    """
    Rotate a square tensor without making any extra border.

    Args:
        src (Tensor): square tensor
        angle_factor (float): angle factor in [0.0,1.0). 1.0 means an entire cycle.

    Returns:
        Tensor: rotated square tensor ([C,H,W]=[src,src_size/(sin(a)+cos(a)),H])

    Note:
        `dst is src` (no copy) if there is nothing to do.
    """

    src_size, src_w = src.shape[-2:]
    assert src_size == src_w

    # rotate with high resolution
    dst = rotate_by_factor(src, angle_factor)

    # strip the border
    angle_rad = (angle_factor % 0.25) * (2 * math.pi)
    div_factor = math.sin(angle_rad) + math.cos(angle_rad)
    if div_factor != 1.0:  # sin(pi/2) is identical to 1.0
        crop_t = src_size / div_factor
        dst = F.center_crop(dst, crop_t)

    return dst


def square_resize(src: Tensor, target_size: int = DEFAULT_TARGET_SIZE) -> Tensor:
    """
    Resize a square tensor into another square shape.

    Args:
        src (Tensor): tensor ([C,H,W]=[ud,H,H])
        target_size (int, optional): target size. Defaults to `DEFAULT_TARGET_SIZE`.

    Returns:
        Tensor: tensor ([C,H,W]=[src,target_size,target_size])
    """

    src_size, src_w = src.shape[-2:]
    assert src_size == src_w

    dst = F_t.resize(src, [target_size, target_size], antialias=True)
    return dst


def from_img(src: Tensor, angle_factor: float, target_size: int = DEFAULT_TARGET_SIZE) -> Tensor:
    """
    Generate rotated square tensor from general tensor.

    - crop the tensor into square shape
    - then rotate it without any extra border
    - then resize it to the target size

    Args:
        src (Tensor): tensor (dtype=uint8, range=[0,255])
        angle_factor (float): angle factor in [0.0,1.0). 1.0 means an entire cycle.
        target_size (int, optional): target size. Defaults to `DEFAULT_TARGET_SIZE`.

    Returns:
        Tensor: tensor ([C,H,W]=[src,target_size,target_size], dtype=float32, range=[0.0,1.0))
    """

    dst = to_square(src)
    dst = u8_to_float32(dst)
    dst = rotate_square(dst, angle_factor)
    dst = square_resize(dst, target_size)

    return dst


def strip_border(src: Tensor) -> Tensor:
    """
    Strip the border out of the middle circle.

    Args:
        src (Tensor): square tensor with border

    Returns:
        Tensor: striped tensor ([C,H,W]=[src,src_size/sqrt(2),H])
    """

    src_size, src_w = src.shape[-2:]
    assert src_size == src_w

    dst = F.center_crop(src, src_size / SQRT2)
    return dst


def from_captcha(src: Tensor, angle_factor: float, target_size: int = DEFAULT_TARGET_SIZE) -> Tensor:
    """
    Generate rotated square tensor from square captcha image tensor with border.

    - rotate it without making any extra border
    - then strip the border
    - then resize it into the target size

    Args:
        src (Tensor): square captcha image tensor ([C,H,W]=[ud,H,H], dtype=uint8, range=[0,255])
        angle_factor (float): angle factor in [0.0,1.0). 1.0 leads to an entire cycle.
        target_size (int, optional): target size. Defaults to `DEFAULT_TARGET_SIZE`.

    Returns:
        Tensor: tensor ([C,H,W]=[src,target_size,target_size], dtype=float32, range=[0.0,1.0))
    """

    dst = u8_to_float32(src)
    dst = rotate_by_factor(dst, angle_factor)
    dst = strip_border(dst)
    dst = square_resize(dst, target_size)

    return dst
