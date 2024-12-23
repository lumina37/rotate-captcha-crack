import argparse
from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from rotate_captcha_crack.common import device
from rotate_captcha_crack.criterion import dist_onehot
from rotate_captcha_crack.dataset.midware import DEFAULT_NORM, Rotator, ScalarLabel, path_to_tensor
from rotate_captcha_crack.dataset.paths import glob_imgs
from rotate_captcha_crack.dataset.pipe import SeqSupportsPipe
from rotate_captcha_crack.helper import default_num_workers
from rotate_captcha_crack.model import RotNet, WhereIsMyModel

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", "-i", type=int, default=-1, help="Use which index")
    opts = parser.parse_args()

    with torch.no_grad():
        dataset_root = Path("../data/captcha")

        img_paths = list(glob_imgs(dataset_root))
        cls_num = 360

        test_dataset = img_paths | SeqSupportsPipe() | path_to_tensor | Rotator() | DEFAULT_NORM | ScalarLabel() | tuple
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=128,
            num_workers=default_num_workers(),
            drop_last=True,
        )

        model = RotNet(cls_num=cls_num, train=False)
        model_path = WhereIsMyModel(model).with_index(opts.index).model_dir / "best.pth"
        print(f"Use model: {model_path}")
        model.load_state_dict(torch.load(model_path, weights_only=True))
        model.to(device=device)
        model.eval()

        total_degree_diff = 0.0
        batch_count = 0

        for source, target in test_dataloader:
            source: Tensor = source.to(device=device)
            target: Tensor = target.to(device=device)

            predict: Tensor = model(source)

            digree_diff = dist_onehot(predict, target) * predict.shape[1]
            total_degree_diff += digree_diff

            batch_count += 1

        print(f"test_loss: {total_degree_diff/batch_count:.4f} degrees")
