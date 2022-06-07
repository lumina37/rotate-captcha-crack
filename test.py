import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from PIL import Image
from torchvision import transforms

from config import CONFIG, device
from model import RotationNet
from utils import find_out_model_path

parser = argparse.ArgumentParser()
parser.add_argument("--timestamp", "-ts", type=int, default=0, help="Use which timestamp")
parser.add_argument("--epoch", type=int, default=0, help="Use which epoch")
opts = parser.parse_args()

if __name__ == "__main__":
    img_size: int = CONFIG['dataset']['img_size']
    trans = transforms.Compose(
        [
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    with torch.no_grad():
        model_dir = Path("models")
        model = RotationNet(train=False)
        model_path = find_out_model_path(opts.timestamp, opts.epoch)
        print(f"Use model: {model_path}")
        model.load_state_dict(torch.load(str(model_path), map_location=device))
        model = model.to(device)
        model.eval()
        img = Image.open("datasets/Landscape-Dataset/pytorch/test/724.jpg")

        img_tensor: torch.Tensor = trans(img).unsqueeze_(0).to(device)
        predict: torch.Tensor = model(img_tensor)
        degree: float = predict.cpu().item() * 360
        print(f"Predict degree: {degree:.4f}")

    img = img.rotate(-degree, resample=Image.Resampling.BILINEAR, fillcolor=(255, 255, 255))  # 因为是复原，这里需要-degree
    plt.figure("debug")
    plt.imshow(img)
    plt.show()
