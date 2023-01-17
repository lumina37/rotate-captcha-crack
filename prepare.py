import shutil
from io import BytesIO
from pathlib import Path
from typing import List, Literal, Tuple

import numpy as np
from PIL import Image, ImageDraw
from torch.utils.data.datapipes.utils.decoder import imagehandler
from torchdata.datapipes.iter import FileLister, FileOpener, IterableWrapper, IterDataPipe, RoutedDecoder, Saver, Zipper
from torchvision import transforms

from rotate_captcha_crack import CONFIG, root

filelist_datapipe = FileLister(root=str(root), masks='*.jpg')
filelist_datapipe = list(filelist_datapipe)

# 划分训练集验证集测试集的数量
total_num = len(list(filelist_datapipe))
train_ratio: float = CONFIG['dataset']['train_ratio']
val_ratio: float = CONFIG['dataset']['val_ratio']
test_ratio: float = CONFIG['dataset']['test_ratio']
sum_ratio = train_ratio + val_ratio + test_ratio
train_num = int(total_num * (train_ratio / sum_ratio))
val_num = int(total_num * (val_ratio / sum_ratio))
train_datapipe = filelist_datapipe[:train_num]
val_datapipe = filelist_datapipe[train_num : train_num + val_num]
test_datapipe = filelist_datapipe[train_num + val_num :]

# create mask
img_size: int = CONFIG['dataset']['img_size']
circle_mask = Image.new('L', (img_size, img_size), color=255)  # white background
circle_draw = ImageDraw.Draw(circle_mask)
circle_draw.ellipse((0, 0, img_size, img_size), fill=0)  # black circle in center

# transforms
trans = transforms.Compose(
    [
        transforms.Resize(img_size),
        transforms.RandomCrop(img_size),
    ]
)


def process_datapipe(datapipe: List[str], save_type: Literal["train", "val", "test"]) -> None:

    save_dir = root / "pytorch" / save_type
    if save_dir.exists():
        shutil.rmtree(str(save_dir))
    save_dir.mkdir(mode=0o755, parents=True)

    dp_size = len(datapipe)
    datapipe: IterDataPipe = FileOpener(datapipe, mode='b')  # 打开文件
    datapipe = RoutedDecoder(datapipe, imagehandler("pil"))  # 解码图像

    rand_rot_factors = np.random.random_sample(size=dp_size)  # 生成[0, 1)的旋转系数
    label_datapipe = IterableWrapper(rand_rot_factors, deepcopy=False)  # 包装为datapipe
    datapipe = Zipper(datapipe, label_datapipe)  # 组合datapipe

    def img_process_fn(tup: Tuple[Tuple[str, Image.Image], float]) -> Tuple[str, bytes]:
        (filepath, img), rot_factor = tup

        square_img: Image.Image = trans(img)  # 放缩和随机crop
        square_img = square_img.rotate(rot_factor * 360, resample=Image.Resampling.BILINEAR)  # 旋图
        square_img.paste(circle_mask, mask=circle_mask)  # 加遮罩

        img_bytesio = BytesIO()
        square_img.save(img_bytesio, format='JPEG', quality=95)
        img_bytes = img_bytesio.getvalue()

        return filepath, img_bytes

    datapipe = datapipe.map(fn=img_process_fn)

    def filepath_fn(src_path: str) -> str:
        src_path: Path = Path(src_path)
        dst_path = str(save_dir / src_path.name)
        return dst_path

    # 保存图像文件
    for _ in Saver(datapipe, mode='wb', filepath_fn=filepath_fn):
        pass
    # 保存旋转系数作为标签
    np.save(str(save_dir.parent / f"{save_type}.npy"), rand_rot_factors)


process_datapipe(train_datapipe, "train")
process_datapipe(val_datapipe, "val")
process_datapipe(test_datapipe, "test")
