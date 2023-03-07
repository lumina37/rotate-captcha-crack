import torch.nn as nn
from torch import Tensor
from torchvision import models


class RCCNet_v0_5(nn.Module):
    """
    RCCNet v0.5
    with avgpool

    Args:
        train (bool, optional): True to load the pretrained parameters. Defaults to True.
    """

    def __init__(self, train: bool = True) -> None:
        super(RCCNet_v0_5, self).__init__()

        weights = models.RegNet_Y_8GF_Weights.DEFAULT if train else None
        self.backbone = models.regnet_y_8gf(weights=weights)

        del self.backbone.fc
        self.avgpool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x: Tensor) -> Tensor:
        """
        forward

        Args:
            x (Tensor): img_tensors ([N,C,H,W]=[batch_size,3,224,224], dtype=float32, range=[0.0,1.0))

        Returns:
            Tensor: predict result ([N]=[batch_size], dtype=float32, range=[0.0,1.0))
        """

        x = self.backbone.forward(x)
        x = self.backbone.stem(x)
        x = self.backbone.trunk_output(x)

        x = self.backbone.avgpool(x)
        x = x.flatten(start_dim=1)
        x = self.avgpool(x)

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

        angle_ts = self.forward(img_ts)
        angle = angle_ts.cpu().item()

        return angle
