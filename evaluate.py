import argparse

import torch
from torchvision import transforms

from config import CONFIG, device
from dataset import get_dataloader
from loss import DistanceBetweenAngles
from model import RotationNet
from utils import find_out_model_path

parser = argparse.ArgumentParser()
parser.add_argument("--timestamp", "-ts", type=int, default=0, help="Use which timestamp")
parser.add_argument("--epoch", type=int, default=0, help="Use which epoch")
opts = parser.parse_args()

with torch.no_grad():
    img_size: int = CONFIG['dataset']['img_size']
    trans = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    eval_criterion = DistanceBetweenAngles()

    batch_size = CONFIG['evaluate']['batch_size']
    test_dataloader = get_dataloader("test", batch_size=batch_size, trans=trans)

    model = RotationNet()
    model.load_state_dict(torch.load(str(find_out_model_path(opts.timestamp, opts.epoch)), map_location=device))
    model = model.to(device)
    model.eval()

    total_degree_diff: float = 0
    batch_count: int = 0

    for source, target in test_dataloader:
        source: torch.Tensor = source.to(device)
        target: torch.Tensor = target.to(device)
        predict: torch.Tensor = model(source)

        total_degree_diff += eval_criterion(predict, target).cpu().item() * 360
        batch_count += 1

    print(f"eval_loss: {total_degree_diff/batch_count:.4f} degrees")
