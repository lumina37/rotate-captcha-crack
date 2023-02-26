import os
from pathlib import Path
from typing import Sequence, Tuple, TypeVar

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


def find_out_model_path(timestamp: int = 0, epoch: int = 0) -> Path:
    """
    Use timestamp and epoch to find out the path of model

    Args:
        timestamp (int): target training time
        epoch (int): target epoch

    Returns:
        Path: path to the model
    """

    model_dir = Path("models")

    if not timestamp:
        ts_dirs = list(model_dir.glob("*"))
        if not ts_dirs:
            raise FileNotFoundError(f"{model_dir} is empty")
        ts_dir = ts_dirs[-1]
        timestamp = int(ts_dir.name)
    ts_dir = model_dir / str(timestamp)

    if not epoch:
        model_paths = list(ts_dir.glob("*.pth"))
        if not model_paths:
            raise FileNotFoundError(f"cannot find any .pth in {ts_dir}")
        model_path = model_paths[-1]
        model_name = model_path.name
    else:
        model_name = f"{epoch}.pth"

    model_path = ts_dir / model_name
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
