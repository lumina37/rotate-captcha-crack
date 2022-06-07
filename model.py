import torch
import torch.nn as nn
from torchvision import models


class RotationNet(nn.Module):
    """
    旋转角度预测网络
    输出为[0,1]的旋转系数
    将该系数乘以2 * pi则映射为图像被旋转的弧度，乘以360则映射为图像被旋转的角度
    """

    def __init__(self) -> None:

        super(RotationNet, self).__init__()

        self.backbone = models.regnet_y_1_6gf(pretrained=True)

        fc_channels = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(fc_channels, 1)

        nn.init.normal_(self.backbone.fc.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.backbone.fc.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone.stem(x)
        x = self.backbone.trunk_output(x)

        x = self.backbone.avgpool(x)
        x = x.flatten(start_dim=1)
        x = self.backbone.fc(x)

        x.squeeze_(dim=1)
        return x


if __name__ == "__main__":
    model = RotationNet()
    print(model)
