import torch
from torch import Tensor

from ..const import DEFAULT_CLS_NUM
from .rotr import RotNetR


class QuantRotNetR(RotNetR):
    def __init__(self, cls_num: int = DEFAULT_CLS_NUM, train: bool = True) -> None:
        super().__init__(cls_num, train)

        self.quant = torch.ao.quantization.QuantStub()
        self.dequant = torch.ao.quantization.DeQuantStub()

    def forward(self, x: Tensor) -> Tensor:
        x = self.quant(x)
        x = super().forward(x)
        x = self.dequant(x)

        return x
