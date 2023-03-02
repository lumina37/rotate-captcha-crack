"""
The output is a factor between [0,1].
Multiply it by 360° then you will get the predict rotated degree.
Use rotate(-degree, ...) to recover the image.
"""

from .helper import WhereIsMyModel
from .rcc_net_v0_2 import RCCNet_v0_2
from .rot_net_reg import RotNet_reg
