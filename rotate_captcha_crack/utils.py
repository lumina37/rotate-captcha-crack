import os
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


NUM_WORKERS = None


def default_num_workers() -> int:
    global NUM_WORKERS

    if NUM_WORKERS is None:
        if (cpu_count := os.cpu_count()) is None:
            NUM_WORKERS = 0
        else:
            cpu_count = cpu_count >> 1
            if cpu_count > 2:
                # reserve 2 cores for other apps
                NUM_WORKERS = cpu_count - 2
            else:
                NUM_WORKERS = 0

    return NUM_WORKERS
