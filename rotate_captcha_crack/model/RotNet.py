import torch.nn as nn
from torch import Tensor
from torchvision import models


class RotNet(nn.Module):
    """
    Args:
        train (bool, optional): True to download pretrained model. Defaults to True.

    Note:
        - impl: `rotnet_street_view_resnet50` in https://github.com/d4nst/RotNet
        - paper: https://arxiv.org/abs/1803.07728
    """

    def __init__(self, train: bool = True) -> None:
        super(RotNet, self).__init__()

        weights = models.ResNet50_Weights.DEFAULT if train else None
        self.backbone = models.resnet50(weights=weights, num_classes=360)

        self.softmax = nn.Softmax(360)

    def forward(self, x: Tensor) -> Tensor:
        x = self.backbone.forward(x)
        x = self.softmax(x)

        return x
