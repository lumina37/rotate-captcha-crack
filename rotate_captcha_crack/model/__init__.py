"""
The output is a factor between [0,1].
Multiply it by 360° then you will get the predict rotated degree.
Use rotate(-degree, ...) to recover the image.
"""

from .helper import FindOutModel
from .RCCNet_fc_1 import RCCNet_fc_1
from .RotNet import RotNet
