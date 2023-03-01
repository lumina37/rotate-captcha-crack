import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from .common import device
from .logging import RCCLogger
from .loss import DistanceBetweenAngles
from .model import FindOutModel


class Trainer(object):
    """
    entry class for training

    Args:
        model (Module): support `RCCNet` and `RotNet`
        train_dataloader (DataLoader): dl for training
        val_dataloader (DataLoader): dl for validation
        optmizer (Optimizer): set learning rate
        lr_scheduler (ReduceLROnPlateau): change learning rate by epoches
        loss (Module): compute loss between `predict` and `target`
    """

    def __init__(
        self,
        model: Module,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        optmizer: Optimizer,
        lr_scheduler: ReduceLROnPlateau,
        loss: Module,
    ) -> None:
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optmizer = optmizer
        self.lr_scheduler = lr_scheduler
        self.loss = loss

        self.finder = FindOutModel(model)
        self._log = None

    @property
    def log(self) -> RCCLogger:
        """
        get logger
        """

        if self._log is None:
            self._log = RCCLogger(self.finder.model_dir)
        return self._log

    def train(self, epoches: int) -> None:
        lr_vec = np.empty(epoches, dtype=np.float64)
        train_loss_vec = np.empty(epoches, dtype=np.float64)
        eval_loss_vec = np.empty(epoches, dtype=np.float64)
        best_eval_loss = sys.maxsize

        eval_loss_c = DistanceBetweenAngles()

        start_t = time.perf_counter()

        for epoch_idx in range(epoches):
            self.model.train()
            total_train_loss = 0.0
            steps = 0

            for source, target in self.train_dataloader:
                source: Tensor = source.to(device=device)
                target: Tensor = target.to(device=device)

                self.optmizer.zero_grad()
                predict: Tensor = self.model(source)

                loss: Tensor = self.loss(predict, target)
                loss.backward()

                total_train_loss += loss.cpu().item()

                self.optmizer.step()
                steps += 1

            train_loss = total_train_loss / steps
            train_loss_vec[epoch_idx] = train_loss

            self.lr_scheduler.step(metrics=train_loss)
            lr_vec[epoch_idx] = self.lr_scheduler._last_lr[0]

            self.model.eval()
            total_eval_loss = 0.0
            batch_count = 0
            with torch.no_grad():
                for source, target in self.val_dataloader:
                    source: Tensor = source.to(device=device)
                    target: Tensor = target.to(device=device)

                    predict: Tensor = self.model(source)

                    eval_loss: Tensor = eval_loss_c(predict, target)
                    total_eval_loss += eval_loss.cpu().item() * 360
                    batch_count += 1

            eval_loss = total_eval_loss / batch_count
            eval_loss_vec[epoch_idx] = eval_loss

            self.log.info(
                f"Epoch#{epoch_idx}. time_cost: {time.perf_counter()-start_t:.1f} s. train_loss: {train_loss:.8f}. eval_loss: {eval_loss:.4f} degrees"
            )

            torch.save(self.model.state_dict(), str(self.finder.model_dir / "last.pth"))
            if eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
                torch.save(self.model.state_dict(), str(self.finder.model_dir / "best.pth"))

        x = np.arange(epoches, dtype=np.int16)

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.plot(x, eval_loss_vec)
        fig.savefig(str(self.finder.model_dir / "eval_loss.png"))

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.plot(x, train_loss_vec)
        fig.savefig(str(self.finder.model_dir / "train_loss.png"))

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.plot(x, lr_vec)
        fig.savefig(str(self.finder.model_dir / "lr.png"))
