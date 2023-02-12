import torch
import torch.nn as nn
from torch import Tensor


class DistanceBetweenAngles(nn.Module):
    """
    Only for Evaluate!

    cycle (float): 转过360度对应的系数变化. Default to 1.0.
    """

    def __init__(self, cycle: float = 1.0) -> None:
        super(DistanceBetweenAngles, self).__init__()
        self.cycle = cycle
        self.half_cycle = cycle / 2

    @torch.no_grad()
    def forward(self, predict: Tensor, target: Tensor) -> Tensor:
        predict = predict.fmod(self.cycle)
        target = target.fmod(self.cycle)
        loss_tensor = self.half_cycle - ((predict - target).abs_() - self.half_cycle).abs_()
        loss = loss_tensor.mean()
        return loss


class RotationLoss(nn.Module):
    """
    MSELoss的优化版本 加入余弦修正来缩小旋转系数0和1之间的距离
    """

    def __init__(self, lambda_cos: float = 0.24, exponent: float = 2.0) -> None:
        super(RotationLoss, self).__init__()
        self.lambda_cos = lambda_cos
        self.exponent = exponent

    def forward(self, predict: Tensor, target: Tensor) -> Tensor:
        diff = predict - target
        loss_tensor = ((diff * (torch.pi * 2)).cos_() - 1) * (-self.lambda_cos) + diff.pow_(self.exponent)
        loss = loss_tensor.mean()
        return loss
