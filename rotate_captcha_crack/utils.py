from pathlib import Path


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
            raise FileNotFoundError(f"cannot find any .pth in {ts_dir}")
        model_path = model_paths[-1]
        model_name = model_path.name
    else:
        model_name = f"{epoch}.pth"

    model_path = ts_dir / model_name
    return model_path
