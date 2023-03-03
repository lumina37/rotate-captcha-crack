import argparse
from pathlib import Path

import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

from rotate_captcha_crack.common import device
from rotate_captcha_crack.dataset import ImgSeqFromPaths, RotDataset
from rotate_captcha_crack.model import RotNet_reg
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
    dataset_root = Path("./datasets/Landscape-Dataset")

    img_paths = list(dataset_root.glob('*.jpg'))
    train_img_paths = slice_from_range(img_paths, (0.0, 0.9))
    train_dataset = RotDataset(ImgSeqFromPaths(train_img_paths))
    val_img_paths = slice_from_range(img_paths, (0.9, 0.95))
    val_dataset = RotDataset(ImgSeqFromPaths(val_img_paths))

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

    model = RotNet_reg()
    model = model.to(device)

    lr = 0.01
    momentum = 0.9
    optmizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optmizer, factor=0.2, patience=3)
    loss = CrossEntropyLoss()

    epoches = 128
    steps = 128
    trainer = Trainer(model, train_dataloader, val_dataloader, optmizer, lr_scheduler, loss, epoches, steps)
    ### Custom configuration area ###
    #################################

    if opts.resume is not None:
        trainer.resume(opts.resume)

    trainer.train()

    visualize_train(trainer.finder.model_dir)
