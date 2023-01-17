import os
import random
from pathlib import Path

import numpy as np
import torch
import yaml

with open("config.yaml", 'r', encoding='utf-8') as file:
    CONFIG = yaml.load(file, yaml.SafeLoader)

root = Path(CONFIG['dataset']['root'])

seed = 42 + (67232 + 8094)
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)

if torch.cuda.is_available():
    device = torch.device('cuda')
    torch.backends.cudnn.benchmark = True
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
else:
    device = torch.device('cpu')
