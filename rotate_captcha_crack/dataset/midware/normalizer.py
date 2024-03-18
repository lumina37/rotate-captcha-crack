import dataclasses as dcs
from collections.abc import Callable

from torch import Tensor
from torchvision.transforms import Normalize

from .labels import ImgWithLabel


@dcs.dataclass
class NormWrapper[TLabel]:
    norm: Callable[[Tensor], Tensor]

    def __call__(self, data: ImgWithLabel[TLabel]) -> ImgWithLabel[TLabel]:
        data.img = self.norm(data.img)
        return data


DEFAULT_NORM = NormWrapper(
    Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        inplace=True,
    )
)
