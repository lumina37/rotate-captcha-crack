from torch import Tensor

ONE_CYCLE = 1.0
HALF_CYCLE = ONE_CYCLE / 2


def dist_between_angles(lhs: Tensor, rhs: Tensor) -> float:
    """
    calculate the average distance between two angle array

    Args:
        lhs (Tensor): lhs tensor ([N]=[undefined], dtype=float32, range=[0,1])
        rhs (Tensor): rhs tensor ([N]=[undefined], dtype=float32, range=[0,1])

    Returns:
        float: average distance. range=[0,1]

    Note:
        Multiply it by 360° to obtain dist in degrees.
    """

    lhs = lhs.fmod(ONE_CYCLE)  # need copy
    rhs = rhs.fmod(ONE_CYCLE)

    loss_tensor = lhs.sub_(rhs).abs_().sub_(HALF_CYCLE).abs_().sub_(HALF_CYCLE).neg_()
    # lhs should not be used below !!!

    loss = loss_tensor.mean().cpu().item()
    return loss

