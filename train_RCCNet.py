import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from rotate_captcha_crack.common import device
from rotate_captcha_crack.dataset import ImgSeqFromPaths, RCCDataset
from rotate_captcha_crack.loss import RotationLoss
from rotate_captcha_crack.lr import LR
from rotate_captcha_crack.model import RCCNet_v0_4
from rotate_captcha_crack.trainer import Trainer
from rotate_captcha_crack.utils import default_num_workers, slice_from_range
from rotate_captcha_crack.visualizer import visualize_train

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--resume", "-r", type=int, default=None, help="Resume from which index. -1 leads to the last training process"
    )
    opts = parser.parse_args()

    #################################
    ### Custom configuration area ###
    dataset_root = Path("E:/Dataset/Streetview/data/data")

    img_paths = list(dataset_root.glob('*.jpg'))
    train_img_paths = slice_from_range(img_paths, (0.0, 0.98))
    train_dataset = RCCDataset(ImgSeqFromPaths(train_img_paths))
    val_img_paths = slice_from_range(img_paths, (0.98, 1.0))
    val_dataset = RCCDataset(ImgSeqFromPaths(val_img_paths))

    num_workers = default_num_workers()
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=64,
        num_workers=num_workers,
        drop_last=True,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=64,
        num_workers=num_workers,
        drop_last=True,
    )

    model = RCCNet_v0_4()
    model = model.to(device)

    lr = 0.0004
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=4, min_lr=lr / 1e4)
    lr = LR(lr, scheduler, optimizer).with_val_loss()
    loss = RotationLoss(lambda_cos=0.24, exponent=2)

    epoches = 64
    steps = 128
    trainer = Trainer(model, train_dataloader, val_dataloader, lr, loss, epoches, steps)
    ### Custom configuration area ###
    #################################

    if opts.resume is not None:
        trainer.resume(opts.resume)

    trainer.train()

    visualize_train(trainer.finder.model_dir)
