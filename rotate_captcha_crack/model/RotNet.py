import torch.nn as nn
from torch import Tensor
from torchvision import models


class RotNet(nn.Module):
    """
    Args:
        train (bool, optional): True to load the pretrained parameters. Defaults to True.

    Note:
        - impl: `rotnet_street_view_resnet50` in https://github.com/d4nst/RotNet
        - paper: https://arxiv.org/abs/1803.07728 TODO: Not this
    """

    def __init__(self, train: bool = True) -> None:
        super(RotNet, self).__init__()

        weights = models.ResNet50_Weights.DEFAULT if train else None
        self.backbone = models.resnet50(weights=weights)

        fc_channels = self.backbone.fc.in_features
        del self.backbone.fc
        self.backbone.fc = nn.Linear(fc_channels, 360)

        if train:
            nn.init.normal_(self.backbone.fc.weight, mean=0.0, std=0.01)
            nn.init.zeros_(self.backbone.fc.bias)

        self.softmax = nn.Softmax(360)

    def forward(self, x: Tensor) -> Tensor:
        """
        predict angle factors

        Args:
            x (Tensor): img_tensor ([N,C,H,W]=[batch_size,3,224,224], dtype=float32, range=[0,1])

        Returns:
            Tensor: predict result ([N]=[batch_size], dtype=float32, range=[0,1])
        """

        x = self.backbone.forward(x)
        x = self.softmax(x)

        return x
