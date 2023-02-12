import functools
from typing import Literal, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose

from rotate_captcha_crack.config import CONFIG

mode_to_nptype = {"I": np.int32, "I;16": np.int16, "F": np.float32}


def to_tensor(img: Image.Image, device: torch.device) -> Tensor:
    img = img.convert('RGB')
    img_np = np.array(img, mode_to_nptype.get(img.mode, np.uint8), copy=True)
    img_tensor = torch.from_numpy(img_np).to(device)

    img_tensor = img_tensor.view(img.size[1], img.size[0], 3)
    img_tensor = img_tensor.permute(2, 0, 1)
    if img_tensor.dtype != torch.get_default_dtype():
        img_tensor = img_tensor.to(dtype=torch.get_default_dtype(), copy=False).div(255)

    return img_tensor


class RCCDataset(Dataset[Tuple[Tensor, Tensor]]):
    def __init__(
        self, ds_type: Literal["train", "val", "test"], device: torch.device, trans: Optional[Compose] = None
    ) -> None:
        self.ds_type: str = ds_type
        ds_dir = CONFIG.dataset.root / "pytorch" / ds_type

        self._img_paths = list(ds_dir.glob('*.jpg'))
        self._img_paths.sort(key=lambda p: int(p.stem))

        angles = np.load(ds_dir.parent / f"{ds_type}.npy")
        angles = torch.from_numpy(angles).to(device)
        angles = angles / 360
        self._angles = angles.to(torch.get_default_dtype())

        if trans:

            def img2tensor(img: Image.Image) -> Tensor:
                img_tensor = to_tensor(img, device)
                img_tensor = trans(img_tensor)
                return img_tensor

        else:
            img2tensor = functools.partial(to_tensor, device=device)

        self.__img2tensor = img2tensor

    def __len__(self) -> int:
        return self._img_paths.__len__()

    def __getitem__(self, idx) -> Tuple[Tensor, Tensor]:
        img_path = self._img_paths[idx]
        img = Image.open(img_path)
        img_tensor = self.__img2tensor(img)

        angle = self._angles[idx]

        return img_tensor, angle


def get_dataloader(
    ds_type: Literal["train", "val", "test"],
    batch_size: int,
    device: torch.device,
    trans: Optional[Compose] = None,
    num_workers: int = 0,
) -> DataLoader:
    dataset = RCCDataset(ds_type, device, trans)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=True,
    )

    return dataloader
