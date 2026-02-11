# E:\restormer+volterra\Restormer + Volterra\train_unified_balanced_2000.py
# Unified multi-task training (Balanced by sampling 2000 each per epoch)
# - Rain task: Rain100H + Rain100L combined, sample 2000 each epoch
# - GoPro: sample 2000 each epoch
# - RESIDE: sample 2000 each epoch
# - CSD: sample 2000 each epoch
# => total per epoch = 8000 samples
#
# AMP + GradScaler
# Progressive resize schedule
# Task-wise validation PSNR/SSIM
# Checkpoint: epoch_xxx_ssimAVG_psnrAVG.pth

# E:\restormer+volterra\Restormer + Volterra\train_unified_balanced_2000.py
import os
import sys
import time
import random
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset, ConcatDataset
from torchvision import transforms
from tqdm import tqdm

from torch.amp import autocast, GradScaler
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim


# ----------------------
# Path setup
# ----------------------
CUR_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CUR_DIR)
for p in [CUR_DIR, ROOT_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from models.restormer_volterra import RestormerVolterra
from re_dataset.rain100h_dataset import Rain100HDataset
from re_dataset.rain100l_dataset import Rain100LDataset
from re_dataset.gopro_dataset import GoProDataset
from re_dataset.reside_dataset import RESIDEDataset
from re_dataset.csd_dataset import CSDDataset


# ======================
# Config
# ======================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = True

SAMPLES_PER_TASK = 2000  # 핵심

RAIN100H_TRAIN = r"E:/restormer+volterra/data/rain100H/train"
RAIN100L_TRAIN = r"E:/restormer+volterra/data/rain100L/train"
GOPRO_CSV_TRAIN = r"E:/restormer+volterra/data/GOPRO_Large/gopro_train_pairs.csv"
RESIDE_ROOT = r"E:/restormer+volterra/data/RESIDE-6K"
CSD_ROOT = r"E:/restormer+volterra/data/CSD"

RAIN100H_TEST = r"E:/restormer+volterra/data/rain100H/test"
RAIN100L_TEST = r"E:/restormer+volterra/data/rain100L/test"
GOPRO_CSV_TEST = r"E:/restormer+volterra/data/GOPRO_Large/gopro_test_pairs.csv"

BATCH_SIZE = 1
EPOCHS = 100
LR = 2e-4
NUM_WORKERS = 4
PIN_MEMORY = True

SAVE_DIR = r"E:/restormer+volterra/checkpoints/restormer_volterra_unified_balanced2000"
os.makedirs(SAVE_DIR, exist_ok=True)

resize_schedule = {0: 128, 30: 192, 60: 256}
BASE_SEED = 1234


def get_transform(epoch: int):
    size = 256
    for k in sorted(resize_schedule.keys()):
        if epoch >= k:
            size = resize_schedule[k]
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
    ]), size


def tensor_to_hwc01(x: torch.Tensor) -> np.ndarray:
    if x.dim() == 4:
        x = x[0]
    x = x.detach().float().clamp(0, 1).cpu().numpy()
    return np.transpose(x, (1, 2, 0))


def sample_subset(dataset: Dataset, k: int, seed: int) -> Subset:
    rng = random.Random(seed)
    n = len(dataset)
    if n >= k:
        idxs = rng.sample(range(n), k)
    else:
        idxs = [rng.randrange(n) for _ in range(k)]
    return Subset(dataset, idxs)


def build_balanced_train_dataset(transform, epoch_num: int):
    seed = BASE_SEED + epoch_num

    r100h = Rain100HDataset(root_dir=RAIN100H_TRAIN, transform=transform)
    r100l = Rain100LDataset(root_dir=RAIN100L_TRAIN, transform=transform)
    rain_all = ConcatDataset([r100h, r100l])

    gopro = GoProDataset(GOPRO_CSV_TRAIN, transform=transform)
    reside = RESIDEDataset(root_dir=RESIDE_ROOT, split="train", transform=transform, strict=True)
    csd = CSDDataset(root_dir=CSD_ROOT, split="train", transform=transform)

    rain_sub = sample_subset(rain_all, SAMPLES_PER_TASK, seed + 1)
    gopro_sub = sample_subset(gopro, SAMPLES_PER_TASK, seed + 2)
    reside_sub = sample_subset(reside, SAMPLES_PER_TASK, seed + 3)
    csd_sub = sample_subset(csd, SAMPLES_PER_TASK, seed + 4)

    train_dataset = ConcatDataset([rain_sub, gopro_sub, reside_sub, csd_sub])
    return train_dataset


def main():
    torch.backends.cudnn.benchmark = True
    model = RestormerVolterra().to(DEVICE)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scaler = GradScaler(device="cuda")

    for epoch in range(EPOCHS):
        epoch_num = epoch + 1
        transform, size = get_transform(epoch_num)

        train_dataset = build_balanced_train_dataset(transform, epoch_num)
        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKERS,
            pin_memory=(PIN_MEMORY and DEVICE.type == "cuda"),
            drop_last=True,
        )

        print(f"\n[Epoch {epoch_num}/{EPOCHS}] Unified Balanced (2000/task), input={size}x{size}")

        model.train()
        loss_sum, psnr_sum, ssim_sum, count = 0.0, 0.0, 0.0, 0
        loop = tqdm(train_loader, desc=f"Train [{epoch_num}/{EPOCHS}]", leave=False)

        for inp, gt in loop:
            inp = inp.to(DEVICE, non_blocking=True)
            gt = gt.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast(device_type="cuda", enabled=(USE_AMP and DEVICE.type == "cuda")):
                out = model(inp)
                loss = criterion(out, gt)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            out_np = tensor_to_hwc01(out)
            gt_np = tensor_to_hwc01(gt)

            psnr = compute_psnr(gt_np, out_np, data_range=1.0)
            ssim = compute_ssim(gt_np, out_np, channel_axis=2, data_range=1.0, win_size=7)

            loss_sum += loss.item()
            psnr_sum += psnr
            ssim_sum += ssim
            count += 1

            loop.set_postfix(
                loss=f"{loss.item():.4f}",
                psnr=f"{psnr:.2f}",
                ssim=f"{ssim:.3f}"
            )

        print(f"[Train] loss={loss_sum/count:.6f} | psnr={psnr_sum/count:.2f} | ssim={ssim_sum/count:.4f}")

        ckpt_path = os.path.join(
            SAVE_DIR,
            f"epoch_{epoch_num:03d}_ssim{ssim_sum/count:.4f}_psnr{psnr_sum/count:.2f}.pth"
        )
        torch.save(model.state_dict(), ckpt_path)
        print(f"[Save] {ckpt_path}")


if __name__ == "__main__":
    main()
