from torch.utils.data import Dataset
from torchvision.transforms import Normalize

from ..helper import DEFAULT_NORM, generate_angles, square_and_rotate
from .typing import TypeImgSeq, TypeRCCItem


class RCCDataset(Dataset[TypeRCCItem]):
    """
    dataset for RCCNet

    Args:
        imgseq (TypeImgSeq): upstream dataset
        angle_num (int, optional): how many rotate angles. 4 leads to [0°, 90°, 180°, 270°]. Defaults to 8.
        copy_num (int, optional): how many copies for one img. should be smaller than `angle_num`. Defaults to 4.
        target_size (int, optional): output img size
        norm (Normalize, optional): normalize policy

    Methods:
        - `def __len__(self) -> int:` length of the dataset
        - `def __getitem__(self, idx: int) -> Tensor:` get square img_tensor ([C,H,W]=[3,target_size,target_size], dtype=float32, range=[0,1])
    """

    __slots__ = [
        'imgseq',
        'target_size',
        'angle_num',
        'norm',
        'size',
        'angles',
    ]

    def __init__(
        self,
        imgseq: TypeImgSeq,
        angle_num: int = 8,
        copy_num: int = 4,
        target_size: int = 224,
        norm: Normalize = DEFAULT_NORM,
    ) -> None:
        self.imgseq = imgseq
        self.angle_num = angle_num
        self.copy_num = copy_num if copy_num < angle_num else angle_num
        self.target_size = target_size
        self.norm = norm

        ori_size = self.imgseq.__len__()
        self.size = ori_size * self.copy_num
        self.angles = generate_angles(ori_size, angle_num, copy_num)

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> TypeRCCItem:
        img_idx = idx // self.copy_num
        img_ts = self.imgseq[img_idx]
        angle_ts = self.angles[idx]

        img_ts = square_and_rotate(img_ts, self.target_size, angle_ts.item())
        img_ts = self.norm(img_ts)

        return img_ts, angle_ts
