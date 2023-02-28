import argparse
import math

import matplotlib.pyplot as plt
import torch
from PIL import Image
from torch import Tensor
from torchvision.transforms import functional as F

from rotate_captcha_crack.common import device
from rotate_captcha_crack.dataset.helper import DEFAULT_NORM
from rotate_captcha_crack.model import RCCNet
from rotate_captcha_crack.utils import find_out_model_path

parser = argparse.ArgumentParser()
parser.add_argument("--index", "-i", type=int, default=0, help="Use which index")
opts = parser.parse_args()

if __name__ == "__main__":
    with torch.no_grad():
        model = RCCNet(train=False)
        model_path = find_out_model_path(cls_name=model.__class__.__name__, index=opts.index)
        print(f"Use model: {model_path}")
        model.load_state_dict(torch.load(str(model_path)))
        model = model.to(device=device)
        model.eval()

        img = Image.open("datasets/tieba/1615096421.jpg")
        assert img.height == img.width
        img_size = img.height

        img_ts = F.to_tensor(img)
        img_ts = F.center_crop(img_ts, img_size / math.sqrt(2))
        img_ts = F.resize(img_ts, 224)
        img_ts: Tensor = DEFAULT_NORM(img_ts)
        img_ts = img_ts.unsqueeze_(0)
        img_ts = img_ts.to(device=device)

        predict: Tensor = model(img_ts)
        degree: float = predict.cpu().item() * 360
        print(f"Predict degree: {degree:.4f}°")

    img = img.rotate(
        -degree, resample=Image.Resampling.BILINEAR, fillcolor=(255, 255, 255)
    )  # use neg degree to recover the img
    plt.figure("debug")
    plt.imshow(img)
    plt.show()
