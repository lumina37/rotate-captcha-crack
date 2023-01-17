import time
from pathlib import Path

import numpy as np
import torch
from matplotlib import pyplot as plt
from torchvision import transforms

from rotate_captcha_crack import CONFIG, LOG, DistanceBetweenAngles, RotationLoss, RotationNet, device, get_dataloader

batch_size: int = CONFIG['train']['batch_size']
epoches: int = CONFIG['train']['epoches']
steps: int = CONFIG['train']['steps']
lr: float = CONFIG['train']['lr']
lambda_cos: float = CONFIG['train']['loss']['lambda_cos']
exponent: float = CONFIG['train']['loss']['exponent']
t_0: float = CONFIG['train']['lr_scheduler']['T_0']
t_mult: float = CONFIG['train']['lr_scheduler']['T_mult']
root = Path(CONFIG['dataset']['root'])

start_time = time.time()
start_time_int = int(start_time)

model_dir = Path(f"models/{start_time_int}")
if not model_dir.exists():
    model_dir.mkdir(parents=True)

trans = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
train_dataloader = get_dataloader("train", batch_size=batch_size, trans=trans, need_shuffle=True)
val_dataloader = get_dataloader("val", batch_size=batch_size, trans=trans)

model = RotationNet()
model = model.to(device)
optmizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optmizer, T_0=t_0, T_mult=t_mult, eta_min=lr / 10e3)
criterion = RotationLoss()
eval_criterion = DistanceBetweenAngles()

lr_vec = np.empty(epoches, dtype=np.float64)
train_loss_vec = np.empty(epoches, dtype=np.float64)
eval_loss_vec = np.empty(epoches, dtype=np.float64)

for epoch_idx in range(epoches):
    model.train()
    total_train_loss: float = 0
    for step_idx, (source, target) in enumerate(train_dataloader):
        source: torch.Tensor = source.to(device)
        target: torch.Tensor = target.to(device)

        optmizer.zero_grad()
        predict: torch.Tensor = model(source)
        loss: torch.Tensor = criterion(predict, target)
        loss.backward()
        total_train_loss += loss.cpu().item()
        optmizer.step()

        if step_idx + 1 == steps:
            break

    scheduler.step()
    lr_vec[epoch_idx] = scheduler.get_last_lr()[0]

    train_loss = total_train_loss / steps
    train_loss_vec[epoch_idx] = train_loss

    model.eval()
    total_eval_loss: float = 0
    batch_count: int = 0
    with torch.no_grad():
        for source, target in val_dataloader:
            source: torch.Tensor = source.to(device)
            target: torch.Tensor = target.to(device)
            predict: torch.Tensor = model(source)

            total_eval_loss += eval_criterion(predict, target).cpu().item() * 360
            batch_count += 1

    eval_loss = total_eval_loss / batch_count
    eval_loss_vec[epoch_idx] = eval_loss

    LOG.info(
        f"Epoch#{epoch_idx}. time_cost: {time.time()-start_time:.2f} s. train_loss: {train_loss:.8f}. eval_loss: {eval_loss:.4f} degrees"
    )

    if epoch_idx >= epoches / 2:
        torch.save(model.state_dict(), str(model_dir / f"{epoch_idx}.pth"))

x = np.arange(epoches, dtype=np.int16)

fig, ax = plt.subplots(figsize=(8, 8))
ax.plot(x, eval_loss_vec)
fig.savefig(str(model_dir / "eval_loss.png"))

fig, ax = plt.subplots(figsize=(8, 8))
ax.plot(x, train_loss_vec)
fig.savefig(str(model_dir / "train_loss.png"))

fig, ax = plt.subplots(figsize=(8, 8))
ax.plot(x, lr_vec)
fig.savefig(str(model_dir / "lr.png"))
