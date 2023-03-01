import torch.nn as nn
from torch import Tensor
from torchvision import models

from ..const import ROTNET_CLS_NUM


class RotNet_reg(nn.Module):
    """
    Args:
        train (bool, optional): True to load the pretrained parameters. Defaults to True.

    Note:
        impl: `rotnet_street_view_resnet50` in https://github.com/d4nst/RotNet but use regnet as backbone
    """

    def __init__(self, train: bool = True) -> None:
        super(RotNet_reg, self).__init__()

        weights = models.RegNet_Y_1_6GF_Weights.DEFAULT if train else None
        self.backbone = models.regnet_y_1_6gf(weights=weights)

        fc_channels = self.backbone.fc.in_features
        del self.backbone.fc
        self.backbone.fc = nn.Linear(fc_channels, ROTNET_CLS_NUM)

        if train:
            nn.init.normal_(self.backbone.fc.weight, mean=0.0, std=0.01)
            nn.init.zeros_(self.backbone.fc.bias)

        self.softmax = nn.Softmax(ROTNET_CLS_NUM)

    def forward(self, x: Tensor) -> Tensor:
        """
        predict angle factors

        Args:
            x (Tensor): img_tensor ([N,C,H,W]=[batch_size,3,224,224], dtype=float32, range=[0,1])

        Returns:
            Tensor: predict result ([N,C]=[batch_size,ROTNET_CLS_NUM], dtype=float32, range=[0,1])
        """

        x = self.backbone.forward(x)
        x = self.softmax(x)

        return x
