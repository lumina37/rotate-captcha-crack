from collections.abc import Callable

from torch import Tensor
from torchvision.transforms import Normalize

from .labels import ImgWithLabel


def norm_wrapper[TLabel](norm: Callable[[Tensor], Tensor]) -> Callable[[ImgWithLabel[TLabel]], ImgWithLabel[TLabel]]:
    def inner(data: ImgWithLabel[TLabel]) -> ImgWithLabel[TLabel]:
        data.img = norm(data.img)
        return data

    return inner


DEFAULT_NORM = norm_wrapper(
    Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        inplace=True,
    )
)
