from typing import Protocol

from torch import Tensor


class TypeImgSeq(Protocol):
    """
    Methods:
        - `def __len__(self) -> int:` length of the dataset
        - `def __getitem__(self, idx: int) -> Tensor:` get img_tensor ([C,H,W]=[3,ud,ud], dtype=float32, range=[0,1])
    """

    def __len__(self) -> int:
        pass

    def __getitem__(self, idx: int) -> Tensor:
        pass
