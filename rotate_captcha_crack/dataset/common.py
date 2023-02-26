from torch.utils.data import Dataset
from torchvision.transforms import Normalize

from ..helper import DEFAULT_NORM, rand_angles, square_and_rotate
from .typing import TypeImgSeq, TypeRCCItem


class RCCDataset(Dataset[TypeRCCItem]):
    """
    dataset for RCC

    Args:
        getimg (TypeGetImg): upstream dataset
        target_size (TypeGetImg, optional): output img size
        norm (Normalize, optional): normalize policy

    Methods:
        `def __len__(self) -> int:` length of the dataset

        `def __getitem__(self, idx: int) -> Tensor:` get img_tensor ([C,H,W]=[3,target_size,target_size], dtype=float32, range=[0,1])
            square tensor with a circle mask (edge is white)
    """

    __slots__ = [
        'getimg',
        'target_size',
        'norm',
        'angles',
    ]

    def __init__(self, getimg: TypeImgSeq, target_size: int = 224, norm: Normalize = DEFAULT_NORM) -> None:
        self.getimg = getimg
        self.target_size = target_size
        self.norm = norm

        self.angles = rand_angles(len(getimg))

    def __len__(self) -> int:
        return self.getimg.__len__()

    def __getitem__(self, idx: int) -> TypeRCCItem:
        angle = self.angles[idx]
        img_ts = self.getimg[idx]

        img_ts = square_and_rotate(img_ts, self.target_size, angle.item())
        img_ts = self.norm(img_ts)

        return img_ts, angle
