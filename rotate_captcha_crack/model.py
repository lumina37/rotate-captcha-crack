import torch
import torch.nn as nn
from torchvision import models


class RotationNet(nn.Module):
    """
    旋转角度预测网络
    输出为[0,1]的旋转系数
    将该系数乘以2 * pi则映射为图像被旋转的弧度，乘以360则映射为图像被旋转的角度

    Args:
        train (bool, optional): 是否使用训练模式 若为True则会自动下载预训练模型. Default to True.
    """

    def __init__(self, train: bool = True) -> None:
        super(RotationNet, self).__init__()

        weights = models.RegNet_X_1_6GF_Weights.DEFAULT if train else None
        self.backbone = models.regnet_x_1_6gf(weights=weights)

        fc_channels = self.backbone.fc.in_features
        self.fc0 = nn.Linear(fc_channels, fc_channels)
        self.act = nn.LeakyReLU()
        self.fc1 = nn.Linear(fc_channels, 1)
        del self.backbone.fc

        if train:
            nn.init.normal_(self.fc0.weight, mean=0.0, std=0.01)
            nn.init.normal_(self.fc1.weight, mean=0.0, std=0.01)
            nn.init.zeros_(self.fc0.bias)
            nn.init.zeros_(self.fc1.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone.stem(x)
        x = self.backbone.trunk_output(x)

        x = self.backbone.avgpool(x)
        x = x.flatten(start_dim=1)

        x = self.fc0(x)
        x = self.act(x)
        x = self.fc1(x)

        x.squeeze_(dim=1)
        return x
