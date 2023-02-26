import sys
import time
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from . import const
from .common import device
from .logging import RCCLogger
from .loss import DistanceBetweenAngles


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

        self._task_name = None
        self._model_dir = None
        self._logger = None

    @property
    def task_name(self) -> str:
        """
        auto-generated unique task_name

        Returns:
            str: task_name

        Example:
            "230229_22_58_59_001"
            "{date}_{hour}_{min}_{sec}_{id}"
        """

        if self._task_name is None:
            start_dt_str = datetime.now().strftime("%y%m%d_%H_%M_%S")
            dt_perfix_len = len(start_dt_str) + 1

            models_dir = Path(const.MODELS_DIR) / self.model.__class__.__name__
            try:
                *_, last_dir = models_dir.iterdir()
                last_idx_str = last_dir.name[dt_perfix_len:]
                last_idx = int(last_idx_str)
                idx = last_idx + 1
            except ValueError:
                # if model_dir is empty or not exist
                models_dir.mkdir(0o755, parents=True, exist_ok=True)
                idx = 0

            self._task_name = f"{start_dt_str}_{idx:0>3}"

        return self._task_name

    @property
    def model_dir(self) -> Path:
        """
        directory to save model

        Returns:
            Path: model_dir
        """

        if self._model_dir is None:
            self._model_dir = Path(const.MODELS_DIR) / self.model.__class__.__name__ / self.task_name
            self._model_dir.mkdir(0o755, exist_ok=True)
        return self._model_dir

    def get_logger(self) -> RCCLogger:
        if self._logger is None:
            self._logger = RCCLogger(self.model_dir)
        return self._logger

    def train(self, epoches: int) -> None:
        lr_vec = np.empty(epoches, dtype=np.float64)
        train_loss_vec = np.empty(epoches, dtype=np.float64)
        eval_loss_vec = np.empty(epoches, dtype=np.float64)
        best_eval_loss = sys.maxsize

        eval_loss_c = DistanceBetweenAngles()

        log = self.get_logger()

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

            log.info(
                f"Epoch#{epoch_idx}. time_cost: {time.perf_counter()-start_t:.1f} s. train_loss: {train_loss:.8f}. eval_loss: {eval_loss:.4f} degrees"
            )

            torch.save(self.model.state_dict(), str(self.model_dir / "last.pth"))
            if eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
                torch.save(self.model.state_dict(), str(self.model_dir / "best.pth"))

        x = np.arange(epoches, dtype=np.int16)

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.plot(x, eval_loss_vec)
        fig.savefig(str(self.model_dir / "eval_loss.png"))

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.plot(x, train_loss_vec)
        fig.savefig(str(self.model_dir / "train_loss.png"))

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.plot(x, lr_vec)
        fig.savefig(str(self.model_dir / "lr.png"))
