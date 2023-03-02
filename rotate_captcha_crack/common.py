import sys

import torch

if sys.version_info >= (3, 11):
    import tomllib  # noqa: F401
else:
    import tomli as tomllib  # noqa: F401

device = torch.device('cuda')
torch.backends.cudnn.benchmark = True
