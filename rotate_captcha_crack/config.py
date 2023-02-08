import os
import random
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import torch

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

seed = 42 + (67232 + 8094)
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)

if torch.cuda.is_available():
    device = torch.device('cuda')
    torch.backends.cudnn.benchmark = True
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
else:
    device = torch.device('cpu')


class DatasetConfig(object):
    __slots__ = [
        '_root',
        '_img_size',
        '_train_ratio',
        '_val_ratio',
        '_test_ratio',
    ]

    def __init__(self, dataset_cfg: dict) -> None:
        self._root: Path = Path(dataset_cfg['root'])
        self._img_size: int = dataset_cfg['img_size']
        self._train_ratio: float = dataset_cfg['train_ratio']
        self._val_ratio: float = dataset_cfg['val_ratio']
        self._test_ratio: float = dataset_cfg['test_ratio']

    @property
    def root(self) -> Path:
        """
        `Path` points to the directory containing pics

        Returns:
            Path

        Note:
            the processed dataset will be placed in {$root}/pytorch/(train or val or test)
        """

        return self._root

    @property
    def img_size(self) -> int:
        """
        img size will be `img_size * img_size` after process

        Returns:
            int

        Note:
            It should fits your model
        """

        return self._img_size

    @property
    def train_ratio(self) -> int:
        """
        how many ratio of imgs will be used for training

        Returns:
            float
        """

        return self._train_ratio

    @property
    def val_ratio(self) -> int:
        """
        how many ratio of imgs will be used for validating

        Returns:
            float
        """

        return self._val_ratio

    @property
    def test_ratio(self) -> int:
        """
        how many ratio of imgs will be used for testing

        Returns:
            float
        """

        return self._test_ratio


class TrainConfig(object):
    __slots__ = [
        'batch_size',
        'epoches',
        'lr',
        '_loss',
        '_lr_scheduler',
    ]

    def __init__(self, train_cfg: dict) -> None:
        self.batch_size: int = train_cfg['batch_size']
        self.epoches: int = train_cfg['epoches']
        self.lr: float = train_cfg['lr']
        self._loss: Dict[str, float] = train_cfg['loss']
        self._lr_scheduler: Dict[str, float] = train_cfg['lr_scheduler']

    @property
    def loss(self) -> Dict[str, float]:
        """
        config for RotationLoss

        Returns:
            Dict[str,float]
        """

        return self._loss

    @property
    def lr_scheduler(self) -> Dict[str, float]:
        """
        config for learning rate scheduler

        Returns:
            Dict[str,float]
        """

        return self._lr_scheduler


class EvalConfig(object):
    __slots__ = ['batch_size']

    def __init__(self, eval_cfg: dict) -> None:
        self.batch_size: int = eval_cfg['batch_size']


class RCCConfig(object):
    __slots__ = [
        '_dataset',
        '_train',
        '_evaluate',
        '_loss',
        '_lr_scheduler',
    ]

    def __init__(self) -> None:
        with open("config.toml", "rb") as f:
            cfg = tomllib.load(f)
        self._dataset = DatasetConfig(cfg['dataset'])
        self._train = TrainConfig(cfg['train'])
        self._eval = EvalConfig(cfg['evaluate'])

    @property
    def dataset(self) -> DatasetConfig:
        """
        config for dataset

        Returns:
            DatasetConfig
        """

        return self._dataset

    @property
    def train(self) -> TrainConfig:
        """
        config for training

        Returns:
            TrainConfig
        """

        return self._train

    @property
    def eval(self) -> EvalConfig:
        """
        config for evaluate

        Returns:
            EvalConfig
        """

        return self._eval


CONFIG = RCCConfig()
