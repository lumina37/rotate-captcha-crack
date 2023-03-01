import torch.nn as nn
from torch import Tensor
from torchvision import models


class RCCNet_fc_1(nn.Module):
    """
    RCCNet with single fc

    Args:
        train (bool, optional): True to load the pretrained parameters. Defaults to True.
    """

    def __init__(self, train: bool = True) -> None:
        super(RCCNet_fc_1, self).__init__()

        weights = models.RegNet_Y_1_6GF_Weights.DEFAULT if train else None
        self.backbone = models.regnet_y_1_6gf(weights=weights)

        fc_channels = self.backbone.fc.in_features
        del self.backbone.fc
        self.backbone.fc = nn.Linear(fc_channels, 1)

        if train:
            nn.init.normal_(self.backbone.fc.weight, mean=0.0, std=0.01)
            nn.init.zeros_(self.backbone.fc.bias)

    def forward(self, x: Tensor) -> Tensor:
        """
        predict angle factors

        Args:
            x (Tensor): img_tensors ([N,C,H,W]=[batch_size,3,224,224], dtype=float32, range=[0,1])

        Returns:
            Tensor: predict result ([N]=[batch_size], dtype=float32, range=[0,1])
        """

        x = self.backbone.stem(x)
        x = self.backbone.trunk_output(x)

        x = self.backbone.avgpool(x)
        x = x.flatten(start_dim=1)
        x = self.backbone.fc(x)

        x.squeeze_(dim=1)
        return x
