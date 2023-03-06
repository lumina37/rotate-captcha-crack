from typing import Sequence, Tuple, TypeVar

from PIL.Image import Image
from torch import Tensor
from torchvision.transforms import Normalize

from .const import DEFAULT_TARGET_SIZE
from .dataset.helper import DEFAULT_NORM, square_resize, strip_border, to_tensor, u8_to_float32

_T = TypeVar('_T')


def slice_from_range(seq: Sequence[_T], _range: Tuple[float, float]) -> Sequence[_T]:
    """
    slice a sequence from given range

    Args:
        seq (Sequence[_T]): parent sequence
        _range (Tuple[float, float]): select which part of the sequence. Use (0.0,0.5) to select the first half

    Returns:
        Sequence[_T]: sliced sequence
    """

    length = len(seq)

    start = int(_range[0] * length)
    assert start >= 0
    end = int(_range[1] * length)
    assert end >= 0

    return seq[start:end]


def process_captcha(img: Image, target_size: int = DEFAULT_TARGET_SIZE, norm: Normalize = DEFAULT_NORM) -> Tensor:
    """
    convert a captcha image into tensor

    Args:
        img (Image): captcha image (square with border)
        target_size (int, optional): target size. Defaults to DEFAULT_TARGET_SIZE.
        norm (Normalize, optional): normalize policy. Defaults to DEFAULT_NORM.

    Returns:
        Tensor: tensor ([C,H,W]=[3,target_size,target_size], dtype=float32, range=[0.0,1.0))
    """

    img = img.convert('RGB')
    img_ts = to_tensor(img)
    img_ts = strip_border(img_ts)
    img_ts = u8_to_float32(img_ts)
    img_ts = square_resize(img_ts, target_size)
    img_ts = norm(img_ts)

    return img_ts
