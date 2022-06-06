import logging
from pathlib import Path
import sys

logging._srcfile = None
logging.logThreads = False
logging.logProcesses = False
logging.logMultiprocessing = False
logging.raiseExceptions = False
logging.Formatter.default_msec_format = '%s.%03d'


class _Logger(logging.Logger):
    """
    自定义的日志记录类

    Args:
        timestamp (int): 启动时的时间戳
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


LOG = None


def get_logger(timestamp: int) -> _Logger:
    global LOG
    if LOG is None:
        LOG = _Logger(timestamp)
    return LOG


def find_out_model_path(timestamp: int = 0, epoch: int = 0) -> Path:
    """
    利用timestamp和epoch找到模型文件的路径

    Args:
        timestamp (int): 训练开始的时间戳
        epoch (int): 训练的epoch

    Returns:
        Path: 模型文件路径
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
            raise FileNotFoundError(f"cant find any .pth in {ts_dir}")
        model_path = model_paths[-1]
        model_name = model_path.name
    else:
        model_name = f"{epoch}.pth"

    model_path = ts_dir / model_name
    return model_path
