from pathlib import Path
from typing import List


def from_google_streetview(root: Path) -> List[Path]:
    """
    get image paths from Google StreetView Dataset

    Args:
        root (Path):  image directory

    Returns:
        list[Path]: image paths

    Note:
        Google StreetView Dataset: https://www.crcv.ucf.edu/data/GMCP_Geolocalization/
    """

    # ignore images with markers (0) and upward views (5)
    paths = [p for p in root.glob('*.jpg') if int(p.stem.rsplit('_')[1]) not in [0, 5]]
    return paths
