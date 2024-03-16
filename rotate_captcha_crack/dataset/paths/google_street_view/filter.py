from pathlib import Path


def filter(path: Path) -> Path:
    """
    Filter the path with viewid = 0 (duplicated main view) or 5 (upward view).

    Args:
        path (Path): image path

    Returns:
        Path: image path

    Note:
        Return None if the image path is been filtered
    """

    segs = path.stem.rsplit('_')
    viewid = int(segs[1])
    return None if viewid in [0, 5] else path
