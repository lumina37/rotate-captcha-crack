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

        weights = models.RegNet_Y_3_2GF_Weights.DEFAULT if train else None
        self.backbone = models.regnet_y_3_2gf(weights=weights)

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
        """
        Args:
            x (Tensor): img_tensors ([N,C,H,W]=[batch_size,3,224,224], dtype=float32, range=[0.0,1.0))

        Returns:
            Tensor: predict result ([N]=[batch_size], dtype=float32, range=[0.0,1.0))
        """

        x = self.backbone.stem(x)
        x = self.backbone.trunk_output(x)

        x = self.backbone.avgpool(x)
        x = x.flatten(start_dim=1)
        x = self.fc0(x)
        x = self.act(x)
        x = self.fc1(x)

        x.squeeze_(1)

        return x

    def predict(self, img_ts: Tensor) -> float:
        """
        Predict the counter clockwise rotation angle.

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
