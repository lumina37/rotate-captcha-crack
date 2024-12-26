import sys

import torch

if sys.version_info >= (3, 11):
    import tomllib  # noqa: F401
else:
    import tomli as tomllib  # noqa: F401

# Use last CUDA, since `cuda:0` is always filled with tasks
device = torch.device('cuda', torch.cuda.device_count() - 1)
torch.backends.cudnn.benchmark = True
