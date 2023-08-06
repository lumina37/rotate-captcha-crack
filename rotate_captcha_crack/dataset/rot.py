from typing import Tuple

import torch
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms import Normalize

from ..const import DEFAULT_CLS_NUM, DEFAULT_TARGET_SIZE
from .helper import DEFAULT_NORM, from_img
from .typing import TypeImgTsSeq

TypeRotItem = Tuple[Tensor, Tensor]


class RotDataset(Dataset[TypeRotItem]):
    """
    Dataset for RotNet (classification).

    Args:
        imgseq (TypeImgSeq): upstream dataset
        target_size (int, optional): output img size. Defaults to `DEFAULT_TARGET_SIZE`.
        norm (Normalize, optional): normalize policy. Defaults to `DEFAULT_NORM`.

    Methods:
        - `def __len__(self) -> int:` length of the dataset
        - `def __getitem__(self, idx: int) -> TypeRotItem:` get square img_ts and index_ts
            ([C,H,W]=[3,target_size,target_size], dtype=float32, range=[0.0,1.0)), ([N]=[1], dtype=long, range=[0,cls_num))
    """

    __slots__ = [
        'imgseq',
        'cls_num',
        'target_size',
        'norm',
        'size',
        'indices',
    ]

    def __init__(
        self,
        imgseq: TypeImgTsSeq,
        cls_num: int = DEFAULT_CLS_NUM,
        target_size: int = DEFAULT_TARGET_SIZE,
        norm: Normalize = DEFAULT_NORM,
    ) -> None:
        self.imgseq = imgseq
        self.cls_num = cls_num
        self.target_size = target_size
        self.norm = norm

        self.size = self.imgseq.__len__()
        self.indices = torch.randint(cls_num, (self.size,), dtype=torch.long)

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> TypeRotItem:
        img_ts = self.imgseq[idx]
        index_ts: Tensor = self.indices[idx]

        img_ts = from_img(img_ts, index_ts.item() / self.cls_num, self.target_size)
        img_ts = self.norm(img_ts)

        return img_ts, index_ts
