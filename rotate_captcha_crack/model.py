import torch.nn as nn
from torch import Tensor
from torchvision import models


class RotationNet(nn.Module):
    """
    Args:
        train (bool, optional): True to download pretrained model. Defaults to True.

    Note:
        The output is a factor between [0,1].
        Multiply it by 360° then you will get the predict rotated degree.
        Use rotate(-degree, ...) to recover the image.
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

    def forward(self, x: Tensor) -> Tensor:
        x = self.backbone.stem(x)
        x = self.backbone.trunk_output(x)

        x = self.backbone.avgpool(x)
        x = x.flatten(start_dim=1)

        x = self.fc0(x)
        x = self.act(x)
        x = self.fc1(x)

        x.squeeze_(dim=1)
        return x
