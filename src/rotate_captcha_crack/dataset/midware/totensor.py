from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import Tensor


def pil_to_tensor(img: Image.Image) -> Tensor:
    """
    Convert PIL image to tensor of type `C3U8`.

    Args:
        img (Image): PIL image

    Returns:
        Tensor: (dtype=uint8, range=[0,255])
    """

    img = img.convert('RGB')
    img_ts = torch.from_numpy(np.array(img))
    img_ts = img_ts.view(img.height, img.width, 3)
    img_ts = img_ts.permute(2, 0, 1)
    return img_ts


def path_to_tensor(path: Path) -> Tensor:
    """
    Read image from path, then convert it to Tensor

    Args:
        path (Path): image path

    Returns:
        Tensor: [C,H,W]=[3,ud,ud], dtype=uint8, range=[0,255]
    """

    img = Image.open(path, formats=('JPEG', 'PNG', 'WEBP'))
    img_ts = pil_to_tensor(img)
    return img_ts


def u8_to_float32(src: Tensor) -> Tensor:
    """
    Convert the dtype of tensor from uint8 to float32.

    Args:
        src (Tensor): tensor (dtype=uint8, range=[0,255])

    Returns:
        Tensor: tensor ([C,H,W]=[src,src,src], dtype=float32, range=[0.0,1.0))
    """

    dst = src.to(dtype=torch.float32).div_(255)
    return dst
