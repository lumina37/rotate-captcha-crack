import torch
from torch.utils.data import Dataset
from torchvision.transforms import Normalize

from .helper import DEFAULT_NORM, square_and_rotate
from .typing import TypeImgSeq, TypeRCCItem


class RCCDataset(Dataset[TypeRCCItem]):
    """
    dataset for RCCNet

    Args:
        imgseq (TypeImgSeq): upstream dataset
        target_size (int, optional): output img size. Defaults to 224.
        norm (Normalize, optional): normalize policy. Defaults to DEFAULT_NORM.

    Methods:
        - `def __len__(self) -> int:` length of the dataset
        - `def __getitem__(self, idx: int) -> Tensor:` get square img_tensor ([C,H,W]=[3,target_size,target_size], dtype=float32, range=[0,1])
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
        target_size: int = 224,
        norm: Normalize = DEFAULT_NORM,
    ) -> None:
        self.imgseq = imgseq
        self.target_size = target_size
        self.norm = norm

        self.size = self.imgseq.__len__()
        self.angles = torch.rand(self.size, dtype=torch.float32)

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> TypeRCCItem:
        img_ts = self.imgseq[idx]
        angle_ts = self.angles[idx]

        img_ts = square_and_rotate(img_ts, angle_ts.item(), self.target_size)
        img_ts = self.norm(img_ts)

        return img_ts, angle_ts
