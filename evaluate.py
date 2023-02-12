import argparse

import torch
from torch import Tensor
from torchvision import transforms

from rotate_captcha_crack import CONFIG, DistanceBetweenAngles, RotationNet, device, find_out_model_path, get_dataloader

parser = argparse.ArgumentParser()
parser.add_argument("--timestamp", "-ts", type=int, default=0, help="Use which timestamp")
parser.add_argument("--epoch", type=int, default=0, help="Use which epoch")
opts = parser.parse_args()

with torch.no_grad():
    img_size = CONFIG.dataset.img_size
    trans = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    eval_criterion = DistanceBetweenAngles()

    batch_size = CONFIG.eval.batch_size
    test_dataloader = get_dataloader("test", batch_size, device, trans)

    model = RotationNet(train=False)
    model_path = find_out_model_path(opts.timestamp, opts.epoch)
    print(f"Use model: {model_path}")
    model.load_state_dict(torch.load(str(model_path), map_location=device))
    model = model.to(device)
    model.eval()

    total_degree_diff = 0.0
    batch_count = 0

    for source, target in test_dataloader:
        predict: Tensor = model(source)
        total_degree_diff += eval_criterion(predict, target).cpu().item() * 360
        batch_count += 1

    print(f"eval_loss: {total_degree_diff/batch_count:.4f} degrees")
