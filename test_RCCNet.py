import argparse
from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from rotate_captcha_crack.common import device
from rotate_captcha_crack.criterion import dist_between_angles
from rotate_captcha_crack.dataset import ImgSeqFromPaths, ValDataset
from rotate_captcha_crack.helper import default_num_workers
from rotate_captcha_crack.model import RCCNet_v0_4, WhereIsMyModel

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", "-i", type=int, default=-1, help="Use which index")
    opts = parser.parse_args()

    with torch.no_grad():
        dataset_root = Path("./datasets/use_img")

        img_paths = list(dataset_root.glob('*.png'))
        test_dataset = ValDataset(ImgSeqFromPaths(img_paths))
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=128,
            num_workers=default_num_workers(),
            drop_last=True,
        )

        model = RCCNet_v0_4(train=False)
        model_path = WhereIsMyModel(model).with_index(opts.index).model_dir / "best.pth"
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

            digree_diff = dist_between_angles(predict, target) * 360
            total_degree_diff += digree_diff

            batch_count += 1

        print(f"test_loss: {total_degree_diff/batch_count:.4f} degrees")
