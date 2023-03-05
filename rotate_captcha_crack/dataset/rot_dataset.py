from typing import Tuple

import torch
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms import Normalize

from ..const import ROTNET_CLS_NUM
from .helper import DEFAULT_NORM, square_and_rotate
from .typing import TypeImgSeq

TypeRotItem = Tuple[Tensor, Tensor]


class RotDataset(Dataset[TypeRotItem]):
    """
    dataset for RotNet

    Args:
        imgseq (TypeImgSeq): upstream dataset
        target_size (int, optional): output img size. Defaults to 224.
        norm (Normalize, optional): normalize policy. Defaults to DEFAULT_NORM.

    Methods:
        - `def __len__(self) -> int:` length of the dataset
        - `def __getitem__(self, idx: int) -> TypeRotItem:` get square img_ts and index_ts\n
            ([C,H,W]=[3,target_size,target_size], dtype=float32, range=[0,1)), ([N]=[1], dtype=long, range=[0,ROTNET_CLS_NUM))
    """

    __slots__ = [
        'imgseq',
        'target_size',
        'norm',
        'size',
        'indices',
    ]

    def __init__(
        self,
        imgseq: TypeImgSeq,
        target_size: int = 224,
        norm: Normalize = DEFAULT_NORM,
    ) -> None:
        self.imgseq = imgseq
        self.target_size = target_size
        self.norm = norm

        self.size = self.imgseq.__len__()
        self.indices = torch.randint(ROTNET_CLS_NUM, (self.size,), dtype=torch.long)

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> TypeRotItem:
        img_ts = self.imgseq[idx]
        index_ts: Tensor = self.indices[idx]

        img_ts = square_and_rotate(img_ts, index_ts.item() / ROTNET_CLS_NUM, self.target_size)
        img_ts = self.norm(img_ts)

        return img_ts, index_ts
