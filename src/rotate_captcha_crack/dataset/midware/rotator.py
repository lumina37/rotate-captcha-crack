import dataclasses as dcs
import random

from torch import Tensor

from ...const import DEFAULT_TARGET_SIZE
from .imgproc import from_img
from .labels import ImgWithLabel


@dcs.dataclass
class Rotator:
    """
    Use this to rotate your image tensor.

    Args:
        target_size (int, optional): target size. Defaults to `DEFAULT_TARGET_SIZE`.
        rng (random.Random): random generator. Defaults to `random.Random()`.

    Methods:
        - `self(img_ts: Tensor) -> ImgWithLabel[float]` \\
            `ret.img` is the rotated image tensor ([C,H,W]=[3,target_size,target_size], dtype=float32, range=[0.0,1.0)). \\
            `ret.label` is the corresponding angle factor (float, range=[0.0,1.0)), where 1.0 means an entire cycle.

    Example:
        ```
        rotator = Rotator()
        ret = rotator(img_ts)
        rotated_img_ts, angle_factor = ret
        angle_factor == ret.label
        ```
    """

    target_size: int = DEFAULT_TARGET_SIZE
    rng: random.Random = dcs.field(default_factory=random.Random)

    def __call__(self, img_ts: Tensor) -> ImgWithLabel[float]:
        angle_factor = self.rng.random()
        square_ts = from_img(img_ts, angle_factor, self.target_size)

        data = ImgWithLabel(square_ts, angle_factor)
        return data
