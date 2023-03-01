from datetime import datetime
from pathlib import Path

from torch.nn import Module

from .. import const

DT_SPLIT_CHAR = '_'
DT_FMT_STR = DT_SPLIT_CHAR.join(["%y%m%d", "%H", "%M", "%S"])


class WhereIsMyModel(object):
    """
    help you find out your model

    Args:
        model (Module): the model you wanna use

    Example:
        for existing model: `FindOutModel(model).with_index(opts.index).model_dir`
        for training new model: `FindOutModel(model).model_dir`
    """

    __slots__ = [
        '_model_name',
        '_model_dir',
        '_task_name',
    ]

    def __init__(self, model: Module) -> None:
        self._model_name = model.__class__.__name__

        self._model_dir: Path = None
        self._task_name: str = None

    def with_index(self, task_index: int = -1) -> "WhereIsMyModel":
        """
        init with task index

        Args:
            task_index (Optional[int], optional): index of task. -1 leads to the last index. Defaults to -1.

        Raises:
            FileNotFoundError

        Returns:
            FindOutModel: self

        Note:
            `task_index` is at the end of `task_name`

        Example:
            - task_name = "230229_22_58_59_002"
            - assert task_index == 002
        """

        models_dir = Path(const.MODELS_DIR) / self._model_name

        if task_index == -1:
            try:
                *_, model_dir = models_dir.iterdir()
            except ValueError:
                raise FileNotFoundError(f"{models_dir} is empty")

        else:
            model_dir = None
            for d in models_dir.iterdir():
                _, index_str = d.name.rsplit(DT_SPLIT_CHAR, 1)
                if task_index == int(index_str):
                    model_dir = d
            if model_dir is None:
                raise FileNotFoundError(f"model_dir not exist. index={task_index}")

        self._task_name = model_dir.name
        self._model_dir = model_dir

        return self

    def with_name(self, task_name: str) -> "WhereIsMyModel":
        """
        init with task name

        Args:
            task_name (str): name of task

        Raises:
            FileNotFoundError

        Returns:
            FindOutModel: self

        Example:
            "230229_22_58_59_002" ~ "{date}_{hour}_{min}_{sec}_{task_index}"
        """

        models_dir = Path(const.MODELS_DIR) / self._model_name

        self._task_name = task_name
        self._model_dir = models_dir / task_name

        if not self._model_dir.is_dir():
            raise FileNotFoundError(f"model_dir {self._model_dir} not exist")

        return self

    @property
    def task_name(self) -> str:
        """
        unique task_name

        Returns:
            str: task_name

        Example:
            "230229_22_58_59_002" ~ "{date}_{hour}_{min}_{sec}_{task_index}"

        Note:
            Auto generate if None
        """

        if self._task_name is None:
            start_dt_str = datetime.now().strftime(DT_FMT_STR)
            dt_perfix_len = len(start_dt_str) + 1

            models_dir = Path(const.MODELS_DIR) / self._model_name
            try:
                *_, last_dir = models_dir.iterdir()
                last_index_str = last_dir.name[dt_perfix_len:]
                last_idx = int(last_index_str)
                task_index = last_idx + 1
            except (ValueError, OSError):
                # if model_dir is empty or not exist
                models_dir.mkdir(0o755, parents=True, exist_ok=True)
                task_index = 0

            self._task_name = f"{start_dt_str}{DT_SPLIT_CHAR}{task_index:0>3}"

        return self._task_name

    @property
    def model_dir(self) -> Path:
        """
        directory to save model

        Returns:
            Path: model_dir

        Note:
            Auto generate if None
        """

        if self._model_dir is None:
            self._model_dir = Path(const.MODELS_DIR) / self._model_name / self.task_name
            self._model_dir.mkdir(0o755, exist_ok=True)
        return self._model_dir
