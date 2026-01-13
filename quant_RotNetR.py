import argparse
from pathlib import Path

import torch
from torch.nn import KLDivLoss
from torch.utils.data import DataLoader

from rotate_captcha_crack.const import DEFAULT_CLS_NUM
from rotate_captcha_crack.dataset.midware import DEFAULT_NORM, CircularSmoothLabel, Rotator, path_to_tensor
from rotate_captcha_crack.dataset.paths import glob_imgs
from rotate_captcha_crack.dataset.pipe import SeqSupportsPipe
from rotate_captcha_crack.lr import LRManager
from rotate_captcha_crack.model import QuantRotNetR, RotNetR, WhereIsMyModel
from rotate_captcha_crack.trainer import Trainer
from rotate_captcha_crack.utils import default_num_workers, get_state_dict, slice_from_range

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", "-i", type=int, default=-1, help="Use which index")
    opts = parser.parse_args()

    with torch.no_grad():
        dataset_root = Path("../data/unlabeled2017")

        img_paths = list(glob_imgs(dataset_root))
        cls_num = DEFAULT_CLS_NUM
        labelling = CircularSmoothLabel(cls_num)

        train_img_paths = slice_from_range(img_paths, (0.0, 0.98))
        train_dataset = (
            train_img_paths | SeqSupportsPipe() | path_to_tensor | Rotator() | DEFAULT_NORM | labelling | tuple
        )
        val_img_paths = slice_from_range(img_paths, (0.98, 1.0))
        val_dataset = val_img_paths | SeqSupportsPipe() | path_to_tensor | Rotator() | DEFAULT_NORM | labelling | tuple

        num_workers = default_num_workers()
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=32,
            num_workers=num_workers,
            shuffle=True,
            drop_last=True,
        )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=32,
            num_workers=num_workers,
            drop_last=True,
        )

        model = RotNetR(cls_num=cls_num, train=False)
        model_path = WhereIsMyModel(model).with_index(opts.index).model_dir / "best.pth"
        print(f"Use model: {model_path}")

        quant_model = QuantRotNetR(cls_num=cls_num, train=False)
        quant_model.load_state_dict(get_state_dict(model_path))

        # model.to(device=device)
        quant_model.eval()

        quant_model.qconfig = torch.ao.quantization.get_default_qat_qconfig("x86")
        quant_model = torch.ao.quantization.fuse_modules(quant_model, [["conv", "bn", "relu"]])
        quant_model = torch.ao.quantization.prepare_qat(quant_model.train())

        lr = 0.001
        momentum = 0.9
        epochs = 32
        steps = 128

        pgroups = [], [], []  # optimizer parameter groups
        bn = tuple(v for k, v in torch.nn.__dict__.items() if "Norm" in k)
        for module_name, module in model.backbone.named_modules():
            for param_name, param in module.named_parameters(recurse=False):
                fullname = f"{module_name}.{param_name}" if module_name else param_name
                if "bias" in fullname:
                    # bias (no decay)
                    pgroups[2].append(param)
                elif isinstance(module, bn):
                    # weight (no decay)
                    pgroups[1].append(param)
                else:
                    # weight (with decay)
                    pgroups[0].append(param)

        optimizer = torch.optim.AdamW(pgroups[2], lr=lr, betas=(momentum, 0.999), weight_decay=0.0)
        optimizer.add_param_group({"params": pgroups[0], "weight_decay": 0.0005})  # add g0 with weight_decay
        optimizer.add_param_group({"params": pgroups[1], "weight_decay": 0.0})  # add g1 (BatchNorm2d weights)

        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=lr, pct_start=0.1, epochs=epochs, steps_per_epoch=steps
        )
        lr = LRManager(lr, scheduler, optimizer)
        loss = KLDivLoss()

        trainer = Trainer(quant_model, train_dataloader, val_dataloader, lr, loss, epochs, steps)

        trainer.train()

        quant_model = torch.ao.quantization.convert(quant_model)
        torch.save(quant_model.state_dict(), trainer.finder.model_dir / "quant.pth")
