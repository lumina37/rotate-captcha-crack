import random
import shutil
from pathlib import Path
from typing import List, Literal

import numpy as np
from PIL import Image, ImageDraw
from torchvision import transforms

from rotate_captcha_crack import CONFIG

ds_cfg = CONFIG.dataset
img_paths = list(ds_cfg.root.glob(ds_cfg.glob_suffix))
total_num = len(img_paths)

train_ratio = ds_cfg.train_ratio
val_ratio = ds_cfg.val_ratio
test_ratio = ds_cfg.test_ratio
sum_ratio = train_ratio + val_ratio + test_ratio
train_num = int(total_num * (train_ratio / sum_ratio))
val_num = int(total_num * (val_ratio / sum_ratio))

random.shuffle(img_paths)
train_paths = img_paths[:train_num]
val_paths = img_paths[train_num : train_num + val_num]
test_paths = img_paths[train_num + val_num :]

# create mask
img_size = ds_cfg.img_size
circle_mask = Image.new('L', (img_size, img_size), color=255)  # white background
circle_draw = ImageDraw.Draw(circle_mask)
circle_draw.ellipse((0, 0, img_size, img_size), fill=0)  # black circle in center

trans = transforms.Compose(
    [
        transforms.Resize(img_size),
        transforms.RandomCrop(img_size),
    ]
)

angle_num = ds_cfg.angle_num
unit_angle = 360 / angle_num
angle_prob = np.full(angle_num, 1 / angle_num, dtype=np.float32)


def process_dataset(paths: List[Path], ds_type: Literal["train", "val", "test"]) -> None:
    save_dir = ds_cfg.root / "pytorch" / ds_type
    if save_dir.exists():
        shutil.rmtree(str(save_dir))
    save_dir.mkdir(mode=0o755, parents=True)

    ds_size = len(paths)
    angles = np.random.choice(angle_num, ds_size, p=angle_prob)
    angles = (angles * unit_angle).astype(np.float32)

    for idx, (img_path, angle) in enumerate(zip(paths, angles)):
        img = Image.open(img_path)
        square_img = trans(img)
        square_img = square_img.rotate(angle, resample=Image.Resampling.BILINEAR)
        square_img.paste(circle_mask, mask=circle_mask)
        square_img.save(save_dir / f'{idx}.jpg', format='JPEG', quality=95)

    np.save(save_dir.parent / f"{ds_type}.npy", angles)


process_dataset(train_paths, "train")
process_dataset(val_paths, "val")
process_dataset(test_paths, "test")
