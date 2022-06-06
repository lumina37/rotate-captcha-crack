__all__ = ['get_dataloader']

from typing import Literal, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.datapipes.utils.decoder import imagehandler
from torchdata.datapipes.iter import FileLister, FileOpener, IterableWrapper, RoutedDecoder, Zipper
from torchvision import transforms

from config import root


def get_dataloader(
    load_type: Literal["train", "val", "test"],
    batch_size: int,
    need_shuffle: bool = False,
    trans: Optional[transforms.Compose] = None,
    num_workers: int = 0,
    pin_memory: bool = True,
) -> DataLoader:

    # 例如从./datasets/Landscape-Dataset/pytorch/train文件夹读取图像
    load_dir = root / "pytorch" / load_type
    img_datapipe = FileLister(root=str(load_dir), masks='*.jpg')
    img_datapipe = FileOpener(img_datapipe, mode='b')
    img_datapipe = RoutedDecoder(img_datapipe, imagehandler("torch"))

    def img_process_fn(tup: Tuple[str, torch.Tensor]) -> torch.Tensor:
        _, img_tensor = tup
        if trans:
            img_tensor = trans(img_tensor)
        return img_tensor

    img_datapipe = img_datapipe.map(fn=img_process_fn)

    # 例如从./datasets/Landscape-Dataset/pytorch/train.npy中读取[0,1)的角度标签
    rand_rot_factors = np.load(str(load_dir.parent / f"{load_type}.npy"))
    rand_rot_factors = torch.from_numpy(rand_rot_factors)
    label_datapipe = IterableWrapper(rand_rot_factors, deepcopy=False)

    datapipe = Zipper(img_datapipe, label_datapipe).shuffle(buffer_size=8192)

    dataloader = DataLoader(
        datapipe,
        batch_size=batch_size,
        shuffle=need_shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )

    return dataloader


if __name__ == "__main__":
    print(next(iter(get_dataloader("train", 2, pin_memory=True))))
