import torch

# Use last CUDA, since `cuda:0` is always filled with tasks
device = torch.device("cuda", torch.cuda.device_count() - 1)
torch.backends.cudnn.benchmark = True
