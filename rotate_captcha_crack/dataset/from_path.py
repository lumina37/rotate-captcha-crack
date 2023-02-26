from pathlib import Path
from typing import Sequence

from PIL import Image
from torch import Tensor
from torchvision.transforms import functional as F

from .typing import TypeImgSeq


class ImgSeqFromPaths(TypeImgSeq):
    """
    Args:
        img_paths (Sequence[Path]): sequence of paths. Each path points to an img file

    Methods:
        `def __len__(self) -> int:` length of the dataset

        `def __getitem__(self, idx: int) -> Tensor:` get img_tensor ([C,H,W]=[3,ud,ud], dtype=float32, range=[0,1])
    """

    __slots__ = ['img_paths']

    def __init__(self, img_paths: Sequence[Path]) -> None:
        self.img_paths = img_paths

    def __len__(self) -> int:
        return self.img_paths.__len__()

    def __getitem__(self, idx: int) -> Tensor:
        img_path = self.img_paths[idx]
        img = Image.open(img_path)
        img = img.convert('RGB')
        img_ts = F.to_tensor(img)
        return img_ts
