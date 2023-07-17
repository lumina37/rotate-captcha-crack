import torch
from torch import Tensor

ONE_CYCLE = 1.0
HALF_CYCLE = ONE_CYCLE / 2


def dist_between_angles(lhs: Tensor, rhs: Tensor) -> float:
    """
    Calculate the average distance between two angle array.

    Args:
        lhs (Tensor): lhs tensor ([N]=[undefined], dtype=float32, range=[0.0,1.0))
        rhs (Tensor): rhs tensor ([N]=[undefined], dtype=float32, range=[0.0,1.0))

    Returns:
        float: average distance. range=[0.0,1.0)

    Note:
        Multiply it by 360° to obtain dist in degrees.
    """

    lhs = lhs.fmod(ONE_CYCLE)
    rhs = rhs.fmod(ONE_CYCLE)
    loss_tensor = lhs.sub_(rhs).abs_().sub_(HALF_CYCLE).abs_().sub_(HALF_CYCLE).neg_()
    del lhs

    loss = loss_tensor.mean().cpu().item()
    return loss


def dist_onehot(one_hot: Tensor, angles: Tensor) -> float:
    """
    Calculate the average distance between one-hot array and angle array.

    Args:
        one_hot (Tensor): one_hot tensor ([N,C]=[undefined,cls_num), dtype=float32, range=[0.0,1.0))
        angles (Tensor): angle tensor ([N]=[undefined], dtype=float32, range=[0.0,1.0))

    Returns:
        float: average distance. range=[0.0,1.0)

    Note:
        Multiply it by `cls_num` to obtain dist in degrees.
    """

    cls_num = one_hot.shape[1]
    one_hot_angles = one_hot.argmax(1).to(dtype=torch.float32).div_(cls_num)
    angles = angles.clone()

    loss = dist_between_angles(one_hot_angles, angles)

    return loss
