import torch
from torch import Tensor

ONE_CYCLE = 1.0
HALF_CYCLE = ONE_CYCLE / 2


def dist_between_angles(lhs: Tensor, rhs: Tensor) -> float:
    """
    calculate the average distance between two angle array

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


def dist_between_angles_(lhs: Tensor, rhs: Tensor) -> float:
    """
    (INPLACE!!!) calculate the average distance between two angle array

    Args:
        lhs (Tensor): lhs tensor ([N]=[undefined], dtype=float32, range=[0.0,1.0))
        rhs (Tensor): rhs tensor ([N]=[undefined], dtype=float32, range=[0.0,1.0))

    Returns:
        float: average distance. range=[0.0,1.0)

    Note:
        Multiply it by 360° to obtain dist in degrees.
    """

    lhs = lhs.fmod_(ONE_CYCLE)  # warn: no copy here. lhs is *moved*
    rhs = rhs.fmod_(ONE_CYCLE)
    loss_tensor = lhs.sub_(rhs).abs_().sub_(HALF_CYCLE).abs_().sub_(HALF_CYCLE).neg_()
    del lhs

    loss = loss_tensor.mean().cpu().item()
    return loss


def dist_between_onehot(lhs: Tensor, rhs: Tensor) -> float:
    """
    calculate the average distance between two one-hot array

    Args:
        lhs (Tensor): lhs tensor ([N,C]=[undefined,cls_num), dtype=float32, range=[0.0,1.0))
        rhs (Tensor): rhs tensor ([N]=[undefined], dtype=long, range=[0,cls_num))

    Returns:
        float: average distance. range=[0.0,1.0)

    Note:
        Multiply it by 360° to obtain dist in degrees.
    """

    cls_num = lhs.nelement()
    lhs = lhs.argmax(1).to(dtype=torch.float32).div_(cls_num)
    rhs = rhs.to(dtype=torch.float32).div_(cls_num)

    loss = dist_between_angles_(lhs, rhs)

    return loss
