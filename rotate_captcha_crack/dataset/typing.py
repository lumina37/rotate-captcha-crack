from typing import Protocol

from torch import Tensor


class TypeImgTsSeq(Protocol):
    """
    Methods:
        - `def __len__(self) -> int:` length of the dataset
        - `def __getitem__(self, idx: int) -> Tensor:` get img_tensor ([C,H,W]=[3,ud,ud], dtype=uint8, range=[0,255])
    """

    def __len__(self) -> int:
        pass

    def __getitem__(self, idx: int) -> Tensor:
        pass
