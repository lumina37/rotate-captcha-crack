import argparse
from pathlib import Path
from typing import Iterable

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from rotate_captcha_crack.common import device
from rotate_captcha_crack.dataset import ImgSeqFromPaths, RCCDataset, TypeRCCItem
from rotate_captcha_crack.loss import DistanceBetweenAngles
from rotate_captcha_crack.model import FindOutModel, RCCNet_fc_1
from rotate_captcha_crack.utils import default_num_workers, slice_from_range

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", "-i", type=int, default=None, help="Use which index")
    opts = parser.parse_args()

    with torch.no_grad():
        dataset_root = Path("./datasets/Landscape-Dataset")

        test_criterion = DistanceBetweenAngles()

        img_paths = list(dataset_root.glob('*.jpg'))
        test_img_paths = slice_from_range(img_paths, (0.95, 1.0))
        test_dataset = RCCDataset(ImgSeqFromPaths(test_img_paths))
        test_dataloader: Iterable[TypeRCCItem] = DataLoader(
            test_dataset,
            batch_size=128,
            num_workers=default_num_workers(),
            drop_last=True,
        )

        model = RCCNet_fc_1(train=False)
        model_path = FindOutModel(model).with_index(opts.index).model_dir / "best.pth"
        print(f"Use model: {model_path}")
        model.load_state_dict(torch.load(str(model_path)))
        model.to(device=device)
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
