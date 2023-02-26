import torch
from torch.utils.data import Dataset
from torchvision.transforms import Normalize

from ..helper import DEFAULT_NORM, square_and_rotate
from .typing import TypeImgSeq, TypeRCCItem


class RCCDataset(Dataset[TypeRCCItem]):
    """
    dataset for RCCNet

    Args:
        getimg (TypeGetImg): upstream dataset
        target_size (int, optional): output img size
        angle_num (int, optional): how many rotate angles. 4 leads to [0°, 90°, 180°, 270°]
        norm (Normalize, optional): normalize policy

    Methods:
        `def __len__(self) -> int:` length of the dataset

        `def __getitem__(self, idx: int) -> Tensor:` get square img_tensor ([C,H,W]=[3,target_size,target_size], dtype=float32, range=[0,1])
    """

    __slots__ = [
        'getimg',
        'target_size',
        'angle_num',
        'norm',
        'length',
    ]

    def __init__(
        self, getimg: TypeImgSeq, target_size: int = 224, angle_num: int = 8, norm: Normalize = DEFAULT_NORM
    ) -> None:
        self.getimg = getimg
        self.target_size = target_size
        self.angle_num = angle_num
        self.norm = norm

        self.length = self.getimg.__len__() * self.angle_num

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> TypeRCCItem:
        img_idx = idx // self.angle_num
        img_ts = self.getimg[img_idx]
        angle = (idx - img_idx * self.angle_num) / self.angle_num
        angle_ts = torch.tensor(angle, dtype=torch.float32)

        img_ts = square_and_rotate(img_ts, self.target_size, angle)
        img_ts = self.norm(img_ts)

        return img_ts, angle_ts
