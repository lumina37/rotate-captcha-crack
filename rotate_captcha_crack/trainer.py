import json
import sys
import time

import numpy as np
import torch
from torch import Tensor
from torch.nn import Module
from torch.utils.data import DataLoader

from .common import device
from .const import CKPT_PATH, LOG_PATH
from .logging import RCCLogger
from .lr import TypeLR
from .model import WhereIsMyModel


class Trainer(object):
    """
    entry class for training

    Args:
        model (Module): support `RCCNet` and `RotNet`
        train_dataloader (DataLoader): dl for training
        val_dataloader (DataLoader): dl for validation
        optimizer (Optimizer): set learning rate
        lr_scheduler (ReduceLROnPlateau): change learning rate by epoches
        loss (Module): compute loss between `predict` and `target`
        epoches (int): how many epoches to train
    """

    __slots__ = [
        'model',
        'train_dataloader',
        'val_dataloader',
        'lr',
        'loss',
        'epoches',
        'steps',
        'finder',
        'lr_array',
        'train_loss_array',
        'val_loss_array',
        'best_val_loss',
        'last_epoch',
        't_cost',
        '_log',
        '_is_new_task',
    ]

    def __init__(
        self,
        model: Module,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        lr: TypeLR,
        loss: Module,
        epoches: int,
        steps: int,
    ) -> None:
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.lr = lr
        self.loss = loss
        self.epoches = epoches
        self.steps = steps
        self.finder = WhereIsMyModel(model)

        self._log = None
        self._is_new_task = True

    @property
    def log(self) -> RCCLogger:
        """
        get logger
        """

        if self._log is None:
            self._log = RCCLogger(self.finder.model_dir / LOG_PATH)
        return self._log

    def resume(self, index: int = -1) -> "Trainer":
        """
        resume from index

        Args:
            index (int, optional): resume from which index. -1 leads to the last training process. Defaults to -1.

        Returns:
            Trainer: self
        """

        self._is_new_task = False
        self.finder.with_index(index)
        self.load_checkpoint()
        return self

    def save_checkpoint(self) -> None:
        """
        save checkpoint according to `finder`
        """

        checkpoint_dir = self.finder.model_dir / CKPT_PATH

        torch.save(
            {
                'model': self.model.state_dict(),
                'lr': self.lr.state_dict(),
            },
            checkpoint_dir / "last.ckpt",
        )

        with open(checkpoint_dir / "last.json", 'w', encoding='utf-8') as f:
            json.dump(
                {
                    'best_val_loss': self.best_val_loss,
                    'last_epoch': self.last_epoch,
                    't_cost': self.t_cost,
                },
                f,
                separators=(',', ':'),
            )

        np.save(checkpoint_dir / "lr.npy", self.lr_array)
        np.save(checkpoint_dir / "train_loss.npy", self.train_loss_array)
        np.save(checkpoint_dir / "val_loss.npy", self.val_loss_array)

    def load_checkpoint(self) -> None:
        """
        load checkpoint according to `finder`
        """

        checkpoint_dir = self.finder.model_dir / CKPT_PATH

        state_dict = torch.load(checkpoint_dir / "last.ckpt")
        self.model.load_state_dict(state_dict['model'])
        self.lr.load_state_dict(state_dict['lr'])

        with open(checkpoint_dir / "last.json", 'rb') as f:
            variables = json.load(f)
            self.best_val_loss = variables['best_val_loss']
            self.last_epoch = variables['last_epoch']
            self.t_cost = variables['t_cost']

        self.lr_array = np.load(checkpoint_dir / "lr.npy")
        self.train_loss_array = np.load(checkpoint_dir / "train_loss.npy")
        self.val_loss_array = np.load(checkpoint_dir / "val_loss.npy")

    def train(self) -> None:
        """
        training entry point
        """

        if self._is_new_task:
            self.lr_array = np.empty(self.epoches, dtype=np.float64)
            self.train_loss_array = np.empty(self.epoches, dtype=np.float64)
            self.val_loss_array = np.empty(self.epoches, dtype=np.float64)
            self.best_val_loss = sys.maxsize
            self.last_epoch = 0
            self.t_cost = 0.0

            (self.finder.model_dir / CKPT_PATH).mkdir(0o755, exist_ok=True)
            (self.finder.model_dir / LOG_PATH).mkdir(0o755, exist_ok=True)

        for epoch_idx in range(self.last_epoch + 1, self.epoches):
            start_t = time.perf_counter()

            self.model.train()
            total_train_loss = 0.0
            steps = 0

            for source, target in self.train_dataloader:
                source: Tensor = source.to(device=device)
                target: Tensor = target.to(device=device)

                with self.lr.optim_step():
                    predict: Tensor = self.model(source)
                    loss: Tensor = self.loss(predict, target)
                    loss.backward()

                total_train_loss += loss.cpu().item()
                steps += 1

                if steps >= self.steps:
                    break

            train_loss = total_train_loss / steps
            self.train_loss_array[epoch_idx] = train_loss

            self.model.eval()
            total_val_loss = 0.0
            eval_batch_count = 0
            with torch.no_grad():
                for source, target in self.val_dataloader:
                    source: Tensor = source.to(device=device)
                    target: Tensor = target.to(device=device)

                    predict: Tensor = self.model(source)

                    val_loss: Tensor = self.loss(predict, target)
                    total_val_loss += val_loss.mean().cpu().item()
                    eval_batch_count += 1

            val_loss = total_val_loss / eval_batch_count
            self.val_loss_array[epoch_idx] = val_loss

            self.lr.sched_step(val_loss)
            self.lr_array[epoch_idx] = self.lr.last_lr

            self.t_cost += time.perf_counter() - start_t
            self.log.info(
                f"Epoch#{epoch_idx}. time_cost: {self.t_cost:.2f} s. train_loss: {train_loss:.8f}. val_loss: {val_loss:.8f}"
            )

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                torch.save(self.model.state_dict(), self.finder.model_dir / "best.pth")

            self.last_epoch = epoch_idx
            self.save_checkpoint()
