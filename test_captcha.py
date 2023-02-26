import argparse
import math

import matplotlib.pyplot as plt
import torch
from PIL import Image
from torch import Tensor
from torchvision.transforms import functional as F

from rotate_captcha_crack.common import device
from rotate_captcha_crack.helper import DEFAULT_NORM
from rotate_captcha_crack.model import RCCNet
from rotate_captcha_crack.utils import find_out_model_path

parser = argparse.ArgumentParser()
parser.add_argument("--index", "-i", type=int, default=0, help="Use which index")
opts = parser.parse_args()

if __name__ == "__main__":
    with torch.no_grad():
        model = RCCNet(train=False)

        model_path = find_out_model_path(opts.index)
        print(f"Use model: {model_path}")

        model.load_state_dict(torch.load(str(model_path)))
        model = model.to(device=device)
        model.eval()

        img = Image.open("datasets/tieba/1615096444.jpg")
        assert img.height == img.width
        img_size = img.height

        img_t = F.to_tensor(img)
        img_t = F.center_crop(img_t, img_size / 2 * math.sqrt(2))
        img_t = F.resize(img_t, 224)
        img_t: Tensor = DEFAULT_NORM(img_t)
        img_t = img_t.unsqueeze_(0)
        img_t = img_t.to(device=device)

        predict: Tensor = model(img_t)
        degree: float = predict.cpu().item() * 360
        print(f"Predict degree: {degree:.4f}")

    img = img.rotate(
        -degree, resample=Image.Resampling.BILINEAR, fillcolor=(255, 255, 255)
    )  # use neg degree to recover the img
    plt.figure("debug")
    plt.imshow(img)
    plt.show()
