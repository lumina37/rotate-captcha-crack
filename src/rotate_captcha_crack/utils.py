from __future__ import annotations

import os
from collections.abc import Sequence
from typing import TYPE_CHECKING, TypeVar

import torch
from torch import Tensor, nn

from .const import DEFAULT_TARGET_SIZE
from .dataset.midware import DEFAULT_NORM, NormWrapper, pil_to_tensor, square_resize, strip_border, u8_to_float32

if TYPE_CHECKING:
    from pathlib import Path

    from PIL.Image import Image

TSeq = TypeVar("TSeq", bound=Sequence)


def default_num_workers() -> int:
    if (cpu_count := os.cpu_count()) is None:
        worker_num = 0
    else:
        cpu_count = cpu_count >> 1
        if cpu_count > 1:
            # reserve 1 core for other apps
            worker_num = cpu_count - 1
        else:
            worker_num = 0

    return worker_num


def get_state_dict(path: Path) -> nn.Module:
    return torch.load(path, map_location="cpu", weights_only=True)


def slice_from_range(seq: TSeq, range_: tuple[float, float]) -> TSeq:
    """
    Slice sequence following the given range.

    Args:
        seq (TSeq): parent sequence
        range_ (tuple[float, float]): select which part of the sequence. Use (0.0,0.5) to select the first half

    Returns:
        TSeq: sliced sequence
    """

    length = len(seq)

    start = int(range_[0] * length)
    assert start >= 0
    end = int(range_[1] * length)
    assert end >= 0

    return seq[start:end]


def process_captcha(img: Image, target_size: int = DEFAULT_TARGET_SIZE, norm: NormWrapper = DEFAULT_NORM) -> Tensor:
    """
    Convert captcha image into tensor.

    Args:
        img (Image): captcha image (square with border)
        target_size (int, optional): target size. Defaults to `DEFAULT_TARGET_SIZE`.
        norm (NormWrapper, optional): normalize policy. Defaults to `DEFAULT_NORM`.

    Returns:
        Tensor: tensor ([C,H,W]=[3,target_size,target_size], dtype=float32, range=[0.0,1.0))
    """

    img = img.convert("RGB")
    img_ts = pil_to_tensor(img)
    img_ts = strip_border(img_ts)
    img_ts = u8_to_float32(img_ts)
    img_ts = square_resize(img_ts, target_size)
    img_ts = norm.norm(img_ts)

    return img_ts
