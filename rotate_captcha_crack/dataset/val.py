from typing import Tuple

import torch
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms import Normalize

from ..const import DEFAULT_TARGET_SIZE
from .helper import DEFAULT_NORM, from_captcha
from .typing import TypeImgSeq

TypeValItem = Tuple[Tensor, Tensor]


class ValDataset(Dataset[TypeValItem]):
    """
    dataset for validate

    Args:
        imgseq (TypeImgSeq): upstream dataset
        target_size (int, optional): output img size. Defaults to DEFAULT_TARGET_SIZE.
        norm (Normalize, optional): normalize policy. Defaults to DEFAULT_NORM.

    Methods:
        - `def __len__(self) -> int:` length of the dataset
        - `def __getitem__(self, idx: int) -> TypeRCCItem:` get square img_ts and angle_ts\n
            ([C,H,W]=[3,target_size,target_size], dtype=float32, range=[0.0,1.0)), ([N]=[1], dtype=float32, range=[0.0,1.0))
    """

    __slots__ = [
        'imgseq',
        'target_size',
        'norm',
        'size',
        'angles',
    ]

    def __init__(
        self,
        imgseq: TypeImgSeq,
        target_size: int = DEFAULT_TARGET_SIZE,
        norm: Normalize = DEFAULT_NORM,
    ) -> None:
        self.imgseq = imgseq
        self.target_size = target_size
        self.norm = norm

        self.size = self.imgseq.__len__()
        self.angles = torch.rand(self.size, dtype=torch.float32)

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> TypeValItem:
        img_ts = self.imgseq[idx]
        angle_ts = self.angles[idx]

        img_ts = from_captcha(img_ts, angle_ts.item(), self.target_size)
        img_ts = self.norm(img_ts)

        return img_ts, angle_ts
