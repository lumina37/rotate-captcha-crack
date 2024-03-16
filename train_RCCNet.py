import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from rotate_captcha_crack.common import device
from rotate_captcha_crack.const import DEFAULT_CLS_NUM
from rotate_captcha_crack.dataset import google_street_view
from rotate_captcha_crack.dataset.midware import Rotator, path_to_tensor
from rotate_captcha_crack.helper import default_num_workers
from rotate_captcha_crack.loss import RotationLoss
from rotate_captcha_crack.lr import LRManager
from rotate_captcha_crack.model import RCCNet_v0_5
from rotate_captcha_crack.trainer import Trainer
from rotate_captcha_crack.utils import slice_from_range
from rotate_captcha_crack.visualizer import visualize_train

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--resume", "-r", type=int, default=None, help="Resume from which index. -1 leads to the last training process"
    )
    opts = parser.parse_args()

    #################################
    ### Custom configuration area ###
    dataset_root = Path("D:/Dataset/Streetview/data/data")

    img_paths = google_street_view.get_paths(dataset_root)
    cls_num = DEFAULT_CLS_NUM

    train_img_paths = slice_from_range(img_paths, (0.0, 0.98))
    train_dataset = train_img_paths | path_to_tensor | Rotator(cls_num) | tuple
    val_img_paths = slice_from_range(img_paths, (0.98, 1.0))
    val_dataset = val_img_paths | path_to_tensor | Rotator(cls_num) | tuple

    num_workers = default_num_workers()
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=64,
        num_workers=num_workers,
        shuffle=True,
        drop_last=True,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=64,
        num_workers=num_workers,
        drop_last=True,
    )

    model = RCCNet_v0_5()
    model = model.to(device)

    lr = 0.0004
    epochs = 64
    steps = 128
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, pct_start=0.25, epochs=epochs, steps_per_epoch=steps
    )
    lr = LRManager(lr, scheduler, optimizer)
    loss = RotationLoss(lambda_cos=0.24, exponent=2)

    trainer = Trainer(model, train_dataloader, val_dataloader, lr, loss, epochs, steps)
    ### Custom configuration area ###
    #################################

    if opts.resume is not None:
        trainer.resume(opts.resume)

    trainer.train()

    visualize_train(trainer.finder.model_dir)
