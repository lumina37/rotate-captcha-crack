from __future__ import annotations

from typing import TYPE_CHECKING

from ...pipe import IterSupportsPipe
from ..helper import glob_imgs
from .filter import filter_ggstreet

if TYPE_CHECKING:
    from pathlib import Path


def get_paths(root: Path) -> list[Path]:
    """
    Get image paths from Google StreetView Dataset.

    Args:
        root (Path): dataset directory

    Returns:
        list[Path]: image paths

    Reference:
        Google StreetView Dataset: https://www.crcv.ucf.edu/data/GMCP_Geolocalization/
    """

    iterator = glob_imgs(root, (".jpg",))
    iterator = iterator | IterSupportsPipe() | filter_ggstreet
    sequence = list(iterator)
    return sequence
