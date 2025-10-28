import torch
from torch import Tensor
from torch.nn import Module


class RotationLoss(Module):
    """
    Optimized MSELoss. Including a cosine correction to reduce the distance between 0 and 1. \\
    $$ \\mathcal{L}(dist) = {dist}^{2} + \\lambda_{cos} (1 - \\cos(2\\pi*{dist})) $$
    """

    def __init__(self, lambda_cos: float = 0.24, exponent: float = 2.0) -> None:
        super().__init__()
        self.lambda_cos = lambda_cos
        self.exponent = exponent

    def forward(self, predict: Tensor, target: Tensor) -> Tensor:
        """
        Calculate the loss between `predict` and `target`.

        Args:
            predict (Tensor): ([N]=[batch_size], dtype=float32, range=[0.0,1.0))
            target (Tensor): ([N]=[batch_size], dtype=float32, range=[0.0,1.0))

        Returns:
            Tensor: loss ([N]=[batch_size], dtype=float32, range=[0.0,1.0))
        """

        dist = predict - target
        loss_tensor = (dist * (torch.pi * 2)).cos_().sub_(1.0).mul_(-self.lambda_cos).add_(dist.pow_(self.exponent))
        loss = loss_tensor.mean()
        return loss
