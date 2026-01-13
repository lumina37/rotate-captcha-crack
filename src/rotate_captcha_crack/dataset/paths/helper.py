from collections.abc import Iterable, Iterator
from pathlib import Path


def glob_imgs(root: Path, suffixes: Iterable[str] = (".jpg", ".png")) -> Iterator[Path]:
    """
    Glob all image paths from one directory.

    Args:
        root (Path): image directory
        suffixes (Iterable[str], optional): glob suffixes. Defaults to ('.jpg', '.png').

    Yields:
        Path: image path
    """

    for path in root.glob("*"):
        if not path.name.endswith(suffixes):
            continue
        yield path
