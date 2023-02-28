import os
from pathlib import Path
from typing import Optional, Sequence, Tuple, TypeVar

from . import const

_T = TypeVar('_T')


def slice_from_range(seq: Sequence[_T], _range: Tuple[float, float]) -> Sequence[_T]:
    """
    slice a sequence from given range

    Args:
        seq (Sequence[_T]): parent sequence
        _range (Tuple[float, float]): select which part of the sequence. Use (0.0,0.5) to select the first half

    Returns:
        Sequence[_T]: sliced sequence
    """

    length = len(seq)

    start = int(_range[0] * length)
    assert start >= 0
    end = int(_range[1] * length)
    assert end >= 0

    return seq[start:end]


def find_out_model_path(cls_name: str, index: Optional[int] = None) -> Path:
    """
    Use cls_name and index to find out the path of model

    Args:
        cls_name (Module): name of the model cls
        index (int, optional): use which index. use last model if None. Defaults to None.

    Returns:
        Path: path to the model
    """

    models_dir = Path(const.MODELS_DIR) / cls_name

    if index is None:
        try:
            *_, model_dir = models_dir.iterdir()
        except ValueError:
            raise FileNotFoundError(f"{models_dir} is empty")

    else:
        model_dir = None
        for d in models_dir.iterdir():
            _, index_str = d.name.rsplit('_', 1)
            if index == int(index_str):
                model_dir = d
        if model_dir is None:
            raise FileNotFoundError(f"model_dir not exist. index={index}")

    model_path = model_dir / 'best.pth'
    return model_path


NUM_WORKERS = None


def default_num_workers() -> int:
    global NUM_WORKERS

    if NUM_WORKERS is None:
        if (cpu_count := os.cpu_count()) is None:
            NUM_WORKERS = 0
        else:
            cpu_count = cpu_count >> 1
            if cpu_count > 2:
                # reserve 2 cores for other application
                NUM_WORKERS = cpu_count - 2
            else:
                NUM_WORKERS = 0

    return NUM_WORKERS
