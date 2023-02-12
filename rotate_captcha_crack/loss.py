import torch
import torch.nn as nn
from torch import Tensor


class DistanceBetweenAngles(nn.Module):
    """
    Only for Evaluate!

    Args:
        cycle (float): how much will the factor increase after 0°->360°. Defaults to 1.0.
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
    Optimized MSELoss.
    Including a cosine correction to reduce the distance between 0 and 1.
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
