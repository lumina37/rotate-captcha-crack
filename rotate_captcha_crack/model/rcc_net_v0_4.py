import torch.nn as nn
from torch import Tensor
from torchvision import models


class RCCNet_v0_4(nn.Module):
    """
    RCCNet v0.4
    with single fc layer

    Args:
        train (bool, optional): True to load the pretrained parameters. Defaults to True.
    """

    def __init__(self, train: bool = True) -> None:
        super(RCCNet_v0_4, self).__init__()

        weights = models.RegNet_Y_3_2GF_Weights.DEFAULT if train else None
        self.backbone = models.regnet_y_3_2gf(weights=weights)

        fc_channels = self.backbone.fc.in_features
        del self.backbone.fc
        self.backbone.fc = nn.Linear(fc_channels, 1)

        if train:
            nn.init.normal_(self.backbone.fc.weight, mean=0.0, std=0.01)
            nn.init.zeros_(self.backbone.fc.bias)

    def forward(self, x: Tensor) -> Tensor:
        """
        forward

        Args:
            x (Tensor): img_tensors ([N,C,H,W]=[batch_size,3,224,224], dtype=float32, range=[0.0,1.0))

        Returns:
            Tensor: predict result ([N]=[batch_size], dtype=float32, range=[0.0,1.0))
        """

        x = self.backbone.forward(x)
        x.squeeze_(1)

        return x

    def predict(self, img_ts: Tensor) -> float:
        """
        predict the counter clockwise rotation angle

        Args:
            img_ts (Tensor): img_tensor ([C,H,W]=[3,224,224], dtype=float32, range=[0.0,1.0))

        Returns:
            float: predict result. range=[0.0,1.0)

        Note:
            Counter clockwise. Use Image.rotate(-ret * 360) to recover the image.
        """

        img_ts = img_ts.unsqueeze_(0)

        angle_ts = self.backbone.forward(img_ts)
        angle = angle_ts.cpu().item()

        return angle
