import torch
import torch.nn as nn


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
    def forward(self, predict: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        predict = predict.fmod(self.cycle)
        target = target.fmod(self.cycle)
        loss_tensor = self.half_cycle - ((predict - target).abs_() - self.half_cycle).abs_()
        loss = loss_tensor.mean()
        return loss


class RotationLoss(nn.Module):
    """
    MSELoss的优化版本 加入余弦修正来缩小旋转系数0和1之间的距离
    """

    def __init__(self, lambda_cos: float = 0.25) -> None:
        super(RotationLoss, self).__init__()
        self.lambda_cos = lambda_cos

    def forward(self, predict: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = predict - target
        loss_tensor = (diff * (torch.pi * 2)).cos_() * (-self.lambda_cos) + diff.square_()
        loss = loss_tensor.mean()
        return loss


if __name__ == "__main__":
    loss = DistanceBetweenAngles(1)
    predict = torch.tensor([2.9])
    target = torch.tensor([3.1])
    print(loss(predict, target).item() * 360)
