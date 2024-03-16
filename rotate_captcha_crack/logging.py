import logging
import sys
from pathlib import Path
from typing import Optional

from .const import DEFAULT_LOG_DIR, LOG_FILE_SUFFIX

logging.addLevelName(logging.FATAL, "FATAL")
logging.addLevelName(logging.WARN, "WARN")

logging.raiseExceptions = False
logging.Formatter.default_msec_format = '%s.%03d'


class RCCLogger(logging.Logger):
    """
    Args:
        log_dir (Path, optional): Where to save the log file. Use default dir if None. Defaults to None.
    """

    def __init__(self, log_dir: Optional[Path] = None) -> None:
        script_name = Path(sys.argv[0]).stem
        super().__init__(script_name)

        if isinstance(log_dir, Path):
            log_dir.mkdir(0o755, parents=True, exist_ok=True)
            log_filepath = log_dir / (script_name + LOG_FILE_SUFFIX)

        else:
            log_dir = Path(DEFAULT_LOG_DIR)
            log_dir.mkdir(0o755, parents=True, exist_ok=True)
            log_filepath = (log_dir / script_name).with_suffix(LOG_FILE_SUFFIX)

        file_handler = logging.FileHandler(str(log_filepath), encoding='utf-8')
        stream_handler = logging.StreamHandler(sys.stdout)

        file_handler.setLevel(logging.INFO)
        stream_handler.setLevel(logging.DEBUG)

        formatter = logging.Formatter("<{asctime}> [{levelname}] {message}", style='{')
        file_handler.setFormatter(formatter)
        stream_handler.setFormatter(formatter)

        self.addHandler(file_handler)
        self.addHandler(stream_handler)
