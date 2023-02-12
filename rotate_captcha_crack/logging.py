import logging
import sys
from pathlib import Path

logging.logThreads = False
logging.logMultiprocessing = False
logging.raiseExceptions = False
logging.Formatter.default_msec_format = '%s.%03d'


class RCCLogger(logging.Logger):
    """
    Args:
        timestamp (int)
    """

    def __init__(self, timestamp: int) -> None:
        timestamp_str = str(timestamp)
        super().__init__(timestamp_str)

        script_path = Path(sys.argv[0])
        log_dir = script_path.parent / "models" / timestamp_str
        log_dir.mkdir(0o755, parents=True, exist_ok=True)
        log_filepath = log_dir / f"{script_path.stem}.log"

        file_handler = logging.FileHandler(str(log_filepath), encoding='utf-8')

        stream_handler = logging.StreamHandler(sys.stdout)

        file_handler.setLevel(logging.INFO)
        stream_handler.setLevel(logging.DEBUG)

        formatter = logging.Formatter("<{asctime}> [{levelname}] {message}", style='{')
        file_handler.setFormatter(formatter)
        stream_handler.setFormatter(formatter)

        self.addHandler(file_handler)
        self.addHandler(stream_handler)
