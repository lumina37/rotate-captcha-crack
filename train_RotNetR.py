import argparse
from pathlib import Path

import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision.transforms import Normalize

from rotate_captcha_crack.common import device
from rotate_captcha_crack.const import DEFAULT_CLS_NUM
from rotate_captcha_crack.dataset import google_street_view
from rotate_captcha_crack.dataset.midware import CircularSmoothLabel, NormWrapper, Rotator, path_to_tensor
from rotate_captcha_crack.dataset.pipe import SeqSupportsPipe
from rotate_captcha_crack.helper import default_num_workers
from rotate_captcha_crack.lr import LRManager
from rotate_captcha_crack.model import RotNetR
from rotate_captcha_crack.trainer import Trainer
from rotate_captcha_crack.utils import slice_from_range
from rotate_captcha_crack.visualizer import visualize_train

NORMALIZER = NormWrapper(
    Normalize(
        mean=[0.0, 0.0, 0.0],
        std=[1.0, 1.0, 1.0],
        inplace=True,
    )
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--resume", "-r", type=int, default=None, help="Resume from which index. -1 leads to the last training process"
    )
    opts = parser.parse_args()

    #################################
    ### Custom configuration area ###
    dataset_root = Path("E:/dataset/streetview/data")

    img_paths = google_street_view.get_paths(dataset_root)
    cls_num = DEFAULT_CLS_NUM
    labelling = CircularSmoothLabel(cls_num)

    train_img_paths = slice_from_range(img_paths, (0.0, 0.98))
    train_dataset = train_img_paths | SeqSupportsPipe() | path_to_tensor | Rotator() | NORMALIZER | labelling | tuple
    val_img_paths = slice_from_range(img_paths, (0.98, 1.0))
    val_dataset = val_img_paths | SeqSupportsPipe() | path_to_tensor | Rotator() | NORMALIZER | labelling | tuple

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

    model = RotNetR(cls_num)
    model = model.to(device)

    lr = 0.01
    momentum = 0.9
    epochs = 64
    steps = 128
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, pct_start=0.25, epochs=epochs, steps_per_epoch=steps
    )
    lr = LRManager(lr, scheduler, optimizer)
    loss = CrossEntropyLoss()

    trainer = Trainer(model, train_dataloader, val_dataloader, lr, loss, epochs, steps)
    ### Custom configuration area ###
    #################################

    if opts.resume is not None:
        trainer.resume(opts.resume)

    trainer.train()

    visualize_train(trainer.finder.model_dir)
