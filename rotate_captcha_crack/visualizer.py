import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

from .const import CKPT_PATH, FIGURE_PATH


def visualize_train(model_dir: Path) -> None:
    """
    visualize the training process and save figures

    Args:
        model_dir (int): visualize target
    """

    checkpoint_dir = model_dir / CKPT_PATH

    with open(checkpoint_dir / "last.json", 'r', encoding='utf-8') as f:
        variables = json.load(f)
        last_epoch = variables['last_epoch']

    lr_array = np.load(checkpoint_dir / "lr.npy")
    train_loss_array = np.load(checkpoint_dir / "train_loss.npy")
    val_loss_array = np.load(checkpoint_dir / "val_loss.npy")

    epoches = last_epoch + 1
    x = np.arange(epoches, dtype=np.int16)

    figure_dir = model_dir / FIGURE_PATH
    figure_dir.mkdir(0o755, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(x, lr_array[:epoches])
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel('epoches')
    ax.set_title('lr - epoches')
    fig.savefig(figure_dir / "lr.png")

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(x, train_loss_array[:epoches])
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel('epoches')
    ax.set_title('train_loss - epoches')
    fig.savefig(figure_dir / "train_loss.png")

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(x, val_loss_array[:epoches])
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel('epoches')
    ax.set_title('val_loss - epoches')
    fig.savefig(figure_dir / "val_loss.png")
