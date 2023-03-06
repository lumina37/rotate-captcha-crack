from pathlib import Path
from typing import Sequence

from PIL import Image
from torch import Tensor

from .helper import to_tensor
from .typing import TypeImgTsSeq


class ImgTsSeqFromPath(TypeImgTsSeq):
    """
    Args:
        img_paths (Sequence[Path]): sequence of paths. Each path points to an img file

    Methods:
        - `def __len__(self) -> int:` length of the dataset
        - `def __getitem__(self, idx: int) -> Tensor:` get img_tensor ([C,H,W]=[3,ud,ud], dtype=uint8, range=[0,255], extra=not guaranteed to be continuous)
    """

    __slots__ = ['img_paths']

    def __init__(self, img_paths: Sequence[Path]) -> None:
        self.img_paths = img_paths

    def __len__(self) -> int:
        return self.img_paths.__len__()

    def __getitem__(self, idx: int) -> Tensor:
        img_path = self.img_paths[idx]

        img = Image.open(img_path, formats=('JPEG', 'PNG', 'WEBP', 'TIFF'))
        img_ts = to_tensor(img)

        return img_ts
