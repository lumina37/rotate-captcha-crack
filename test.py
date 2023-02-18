import argparse
from typing import Iterable

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from rotate_captcha_crack.config import CONFIG, device
from rotate_captcha_crack.dataset import GetImgFromPaths, RCCDataset, TypeRCCItem
from rotate_captcha_crack.loss import DistanceBetweenAngles
from rotate_captcha_crack.model import RotationNet
from rotate_captcha_crack.utils import find_out_model_path

parser = argparse.ArgumentParser()
parser.add_argument("--timestamp", "-ts", type=int, default=0, help="Use which timestamp")
parser.add_argument("--epoch", type=int, default=0, help="Use which epoch")
opts = parser.parse_args()

if __name__ == '__main__':
    with torch.no_grad():
        ds_cfg = CONFIG.dataset
        test_cfg = CONFIG.test
        test_criterion = DistanceBetweenAngles()

        batch_size = CONFIG.test.batch_size

        img_paths = list(ds_cfg.root.glob(ds_cfg.glob_suffix))
        start = ds_cfg.train_ratio + ds_cfg.val_ratio
        test_range = (start, start + ds_cfg.test_ratio)
        test_dataset = RCCDataset(GetImgFromPaths(img_paths, test_range))
        test_dataloader: Iterable[TypeRCCItem] = DataLoader(
            test_dataset,
            test_cfg.batch_size,
            num_workers=test_cfg.num_workers,
            drop_last=True,
        )

        model = RotationNet(train=False)
        model_path = find_out_model_path(opts.timestamp, opts.epoch)
        print(f"Use model: {model_path}")
        model.load_state_dict(torch.load(str(model_path), map_location=device))
        model = model.to(device)
        model.eval()

        total_degree_diff = 0.0
        batch_count = 0

        for source, target in test_dataloader:
            source: Tensor = source.to(device=device)
            target: Tensor = target.to(device=device)

            predict: Tensor = model(source)

            digree_diff: Tensor = test_criterion(predict, target)
            total_degree_diff += digree_diff.cpu().item() * 360

            batch_count += 1

        print(f"test_loss: {total_degree_diff/batch_count:.4f} degrees")
