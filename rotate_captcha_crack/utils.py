import os
from typing import Sequence, Tuple, TypeVar

from PIL.Image import Image
from torch import Tensor
from torchvision.transforms import Normalize
from torchvision.transforms import functional as F

from .const import SQRT2
from .dataset.helper import DEFAULT_NORM

_T = TypeVar('_T')


def strip_circle_border(src: Image, target_size: int = 224, norm: Normalize = DEFAULT_NORM) -> Tensor:
    """
    strip the circle border

    Args:
        src (Image): source image
        target_size (int, optional): target size. Defaults to 224.
        norm (Normalize, optional): normalize policy

    Returns:
        Tensor: striped tensor ([C,H,W]=[src,target_size,target_size])
    """

    src_size = src.height
    assert src.height == src.width

    src = src.convert('RGB')
    dst = F.to_tensor(src)
    dst = F.center_crop(dst, src_size / SQRT2)
    dst = F.resize(dst, target_size)
    dst = norm(dst)

    return dst


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


NUM_WORKERS = None


def default_num_workers() -> int:
    global NUM_WORKERS

    if NUM_WORKERS is None:
        if (cpu_count := os.cpu_count()) is None:
            NUM_WORKERS = 0
        else:
            cpu_count = cpu_count >> 1
            if cpu_count > 2:
                # reserve 2 cores for other apps
                NUM_WORKERS = cpu_count - 2
            else:
                NUM_WORKERS = 0

    return NUM_WORKERS
