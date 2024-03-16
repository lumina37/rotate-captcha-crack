from pathlib import Path

from ...pipeline import IteratorRoot, SequenceRoot
from ..helper import glob_imgs
from .filter import filter


def get_paths(root: Path) -> SequenceRoot[Path]:
    """
    Get image paths from Google StreetView Dataset.

    Args:
        root (Path): dataset directory

    Returns:
        SequenceRoot[Path]: image paths

    Reference:
        Google StreetView Dataset: https://www.crcv.ucf.edu/data/GMCP_Geolocalization/
    """

    iterator = glob_imgs(root, ('.jpg',))
    iterator = IteratorRoot(iterator) | filter
    sequence = SequenceRoot(list(iterator))
    return sequence
