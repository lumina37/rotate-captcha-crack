from PIL.Image import Image
from torch import Tensor
from torchvision.transforms import Normalize
from torchvision.transforms import functional as F

from . import const
from .dataset.helper import DEFAULT_NORM


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

    dst = F.to_tensor(src)
    dst = F.center_crop(dst, src_size / const.SQRT2)
    dst = F.resize(dst, target_size)
    dst = norm(dst)

    return dst
