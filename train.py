import time
from pathlib import Path

import torch
from torchvision import transforms

from config import CONFIG, device
from dataset import get_dataloader
from loss import DistanceBetweenAngles, RotationLoss
from model import RotationNet
from utils import get_logger

batch_size: int = CONFIG['train']['batch_size']
epoches: int = CONFIG['train']['epoches']
steps: int = CONFIG['train']['steps']
lr: float = CONFIG['train']['lr']
root = Path(CONFIG['dataset']['root'])

start_time = time.time()
start_time_int = int(start_time)
LOG = get_logger(start_time_int)

model_dir = Path(f"models/{start_time_int}")
if not model_dir.exists():
    model_dir.mkdir(parents=True)

trans = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
train_dataloader = get_dataloader("train", batch_size=batch_size, trans=trans, need_shuffle=True)
val_dataloader = get_dataloader("val", batch_size=batch_size, trans=trans)

model = RotationNet()
model = model.to(device)
optmizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optmizer, T_0=4, T_mult=2)
criterion = RotationLoss(lambda_cos=0.25)
eval_criterion = DistanceBetweenAngles()

for epoch_idx in range(epoches):
    model.train()

    for i_step, (source, target) in enumerate(train_dataloader):
        source: torch.Tensor = source.to(device)
        target: torch.Tensor = target.to(device)

        optmizer.zero_grad()
        predict: torch.Tensor = model(source)
        loss: torch.Tensor = criterion(predict, target)
        loss.backward()
        optmizer.step()
        scheduler.step()

        if i_step + 1 == steps:
            break

    LOG.info(f"Epoch#{epoch_idx} time_cost: {time.time()-start_time:.2f} s")

    model.eval()
    total_degree_diff: float = 0
    batch_count: int = 0
    with torch.no_grad():
        for source, target in val_dataloader:
            source: torch.Tensor = source.to(device)
            target: torch.Tensor = target.to(device)
            predict: torch.Tensor = model(source)

            total_degree_diff += eval_criterion(predict, target).cpu().item() * 360
            batch_count += 1

    LOG.info(f"Epoch#{epoch_idx} eval_loss: {total_degree_diff/batch_count:.4f} degrees")

    if epoch_idx >= epoches / 2:
        torch.save(model.state_dict(), str(model_dir / f"{epoch_idx}.pth"))
