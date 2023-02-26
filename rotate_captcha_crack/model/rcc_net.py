import torch.nn as nn
from torch import Tensor
from torchvision import models


class RCCNet(nn.Module):
    """
    Args:
        train (bool, optional): True to download pretrained model. Defaults to True.

    Note:
        The output is a factor between [0,1].
        Multiply it by 360° then you will get the predict rotated degree.
        Use rotate(-degree, ...) to recover the image.
    """

    def __init__(self, train: bool = True) -> None:
        super(RCCNet, self).__init__()

        weights = models.RegNet_Y_1_6GF_Weights.DEFAULT if train else None
        self.backbone = models.regnet_y_1_6gf(weights=weights)

        fc_channels = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(fc_channels, 1)

        if train:
            nn.init.normal_(self.backbone.fc.weight, mean=0.0, std=0.01)
            nn.init.zeros_(self.backbone.fc.bias)

    def forward(self, x: Tensor) -> Tensor:
        x = self.backbone.stem(x)
        x = self.backbone.trunk_output(x)

        x = self.backbone.avgpool(x)
        x = x.flatten(start_dim=1)
        x = self.backbone.fc(x)

        x.squeeze_(dim=1)
        return x
