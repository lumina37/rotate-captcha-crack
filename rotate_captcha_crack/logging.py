import logging
import sys
from pathlib import Path
from typing import Optional

from . import const

logging.logThreads = False
logging.logMultiprocessing = False
logging.raiseExceptions = False
logging.Formatter.default_msec_format = '%s.%03d'


class RCCLogger(logging.Logger):
    """
    Args:
        save_dir (Path, optional): where to save the log file. use default dir if None. Defaults to None.

    Note:
        make sure the save_dir is already created
    """

    def __init__(self, save_dir: Optional[Path] = None) -> None:
        script_name = Path(sys.argv[0]).stem
        super().__init__(script_name)

        if save_dir:
            assert save_dir.is_dir()
            log_filepath = save_dir / (script_name + ".log")

        else:
            save_dir = Path(const.DEFAULT_LOG_DIR)
            save_dir.mkdir(0o755, parents=True, exist_ok=True)
            log_filepath = (save_dir / script_name).with_suffix(const.LOG_FILE_SUFFIX)

        file_handler = logging.FileHandler(str(log_filepath), encoding='utf-8')
        stream_handler = logging.StreamHandler(sys.stdout)

        file_handler.setLevel(logging.INFO)
        stream_handler.setLevel(logging.DEBUG)

        formatter = logging.Formatter("<{asctime}> [{levelname}] {message}", style='{')
        file_handler.setFormatter(formatter)
        stream_handler.setFormatter(formatter)

        self.addHandler(file_handler)
        self.addHandler(stream_handler)
