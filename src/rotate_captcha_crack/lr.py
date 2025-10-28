import contextlib
from typing import Protocol

from torch.optim.lr_scheduler import _LRScheduler as LRScheduler
from torch.optim.optimizer import Optimizer


class TypeLRManager(Protocol):
    def state_dict(self) -> dict: ...

    def load_state_dict(self, d: dict) -> None: ...

    @property
    def lr(self) -> float:
        """
        original learning rate
        """
        ...

    @property
    def last_lr(self) -> float:
        """
        last learning rate
        """
        ...

    def sched_step(self, val_loss: float) -> None:
        """
        lr_scheduler.step()
        """
        ...

    @contextlib.contextmanager
    def optim_step(self) -> None:
        """
        optimizer.zero_grad()\n
        ...\n
        optimizer.step()
        """
        ...


class LRManagerWithValLoss(TypeLRManager):
    def __init__(self, lr: float, scheduler: LRScheduler, optimizer: Optimizer) -> None:
        self._lr = lr
        self._scheduler = scheduler
        self._optimizer = optimizer

    def state_dict(self) -> dict:
        return {
            "scheduler": self._scheduler.state_dict(),
            "optimizer": self._optimizer.state_dict(),
        }

    def load_state_dict(self, d: dict) -> None:
        self._scheduler.load_state_dict(d['scheduler'])
        self._optimizer.load_state_dict(d['optimizer'])

    @property
    def lr(self) -> float:
        return self._lr

    @property
    def last_lr(self) -> float:
        self._scheduler._last_lr[0]

    def sched_step(self, val_loss: float) -> None:
        self._scheduler.step(metrics=val_loss)

    @contextlib.contextmanager
    def optim_step(self) -> None:
        self._optimizer.zero_grad()
        yield
        self._optimizer.step()


class LRManager(TypeLRManager):
    def __init__(self, lr: float, scheduler: LRScheduler, optimizer: Optimizer) -> None:
        self._lr = lr
        self._scheduler = scheduler
        self._optimizer = optimizer

    def with_val_loss(self) -> LRManagerWithValLoss:
        ret = LRManagerWithValLoss(self._lr, self._scheduler, self._optimizer)
        return ret

    def state_dict(self) -> dict:
        return {
            "scheduler": self._scheduler.state_dict(),
            "optimizer": self._optimizer.state_dict(),
        }

    def load_state_dict(self, d: dict) -> None:
        self._scheduler.load_state_dict(d['scheduler'])
        self._optimizer.load_state_dict(d['optimizer'])

    @property
    def lr(self) -> float:
        return self._lr

    @property
    def last_lr(self) -> float:
        self._scheduler._last_lr[0]

    def sched_step(self, _: float) -> None:
        self._scheduler.step()

    @contextlib.contextmanager
    def optim_step(self) -> None:
        self._optimizer.zero_grad()
        yield
        self._optimizer.step()
