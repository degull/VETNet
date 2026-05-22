# E:\restormer+volterra\Restormer + Volterra\train_unified_balanced_2000.py
# Unified multi-task training (Balanced by sampling 2000 each per epoch)
# - Rain task: Rain100H + Rain100L combined, sample 2000 each epoch
# - GoPro: sample 2000 each epoch
# - RESIDE: sample 2000 each epoch
# - CSD: sample 2000 each epoch
# => total per epoch = 8000 samples
#
# ✅ AMP + GradScaler
# ✅ Progressive resize schedule
# ✅ Train loop: loss/psnr/ssim live 출력(tqdm)
# ✅ Epoch end: Train 평균 loss/psnr/ssim 출력
# ✅ Task-wise Validation: Rain100H, Rain100L, GoPro, RESIDE, CSD PSNR/SSIM 출력
# ✅ Checkpoint: epoch_xxx_ssimAVG_psnrAVG.pth (AVG는 4-task: Rain(avg)+GoPro+RESIDE+CSD)

import os
import sys
import time
import random
from typing import Tuple, Dict

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

SAMPLES_PER_TASK = 2000  # 핵심 (각 task를 2000개로 맞춤)

# ---- paths
RAIN100H_TRAIN = r"E:/restormer+volterra/data/rain100H/train"
RAIN100L_TRAIN = r"E:/restormer+volterra/data/rain100L/train"
GOPRO_CSV_TRAIN = r"E:/restormer+volterra/data/GOPRO_Large/gopro_train_pairs.csv"
RESIDE_ROOT = r"E:/restormer+volterra/data/RESIDE-6K"
CSD_ROOT = r"E:/restormer+volterra/data/CSD"

RAIN100H_TEST = r"E:/restormer+volterra/data/rain100H/test"
RAIN100L_TEST = r"E:/restormer+volterra/data/rain100L/test"
GOPRO_CSV_TEST = r"E:/restormer+volterra/data/GOPRO_Large/gopro_test_pairs.csv"

# ---- training
BATCH_SIZE = 1
EPOCHS = 100
LR = 2e-4
NUM_WORKERS = 4
PIN_MEMORY = True

# ---- save
SAVE_DIR = r"E:/restormer+volterra/checkpoints/restormer_volterra_unified_balanced2000"
os.makedirs(SAVE_DIR, exist_ok=True)

# ---- resize schedule
resize_schedule = {0: 128, 30: 192, 60: 256}
BASE_SEED = 1234


# ======================
# Utils
# ======================
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
    """(1,C,H,W) or (C,H,W) -> HWC float32 [0,1]"""
    if x.dim() == 4:
        x = x[0]
    x = x.detach().float().clamp(0, 1).cpu().numpy()
    return np.transpose(x, (1, 2, 0))


def sample_subset(dataset: Dataset, k: int, seed: int) -> Subset:
    """dataset에서 k개 샘플링. n<k면 replacement로 oversample"""
    rng = random.Random(seed)
    n = len(dataset)
    if n >= k:
        idxs = rng.sample(range(n), k)
    else:
        idxs = [rng.randrange(n) for _ in range(k)]
    return Subset(dataset, idxs)


def build_balanced_train_dataset(transform, epoch_num: int):
    """epoch마다 2000개씩 task-balanced dataset 생성"""
    seed = BASE_SEED + epoch_num

    # Rain = H + L (합쳐서 2000개 샘플)
    r100h = Rain100HDataset(root_dir=RAIN100H_TRAIN, transform=transform)
    r100l = Rain100LDataset(root_dir=RAIN100L_TRAIN, transform=transform)
    rain_all = ConcatDataset([r100h, r100l])

    # GoPro / RESIDE / CSD
    # GoProDataset은 header 포함 CSV면 반드시 header-skip 처리된 버전 권장
    gopro = GoProDataset(GOPRO_CSV_TRAIN, transform=transform)
    reside = RESIDEDataset(root_dir=RESIDE_ROOT, split="train", transform=transform, strict=True)
    csd = CSDDataset(root_dir=CSD_ROOT, split="train", transform=transform, strict=True)

    rain_sub = sample_subset(rain_all, SAMPLES_PER_TASK, seed + 1)
    gopro_sub = sample_subset(gopro, SAMPLES_PER_TASK, seed + 2)
    reside_sub = sample_subset(reside, SAMPLES_PER_TASK, seed + 3)
    csd_sub = sample_subset(csd, SAMPLES_PER_TASK, seed + 4)

    train_dataset = ConcatDataset([rain_sub, gopro_sub, reside_sub, csd_sub])
    counts = {
        "Rain(H+L)": len(rain_sub),
        "GoPro": len(gopro_sub),
        "RESIDE": len(reside_sub),
        "CSD": len(csd_sub),
        "Total": len(train_dataset),
    }
    return train_dataset, counts


def eval_loader(model, loader, name: str) -> Tuple[float, float]:
    """PSNR/SSIM 평균"""
    model.eval()
    psnr_sum, ssim_sum, n = 0.0, 0.0, 0
    with torch.no_grad():
        for inp, gt in tqdm(loader, desc=f"Eval {name}", leave=False):
            inp = inp.to(DEVICE, non_blocking=True)
            gt = gt.to(DEVICE, non_blocking=True)

            with autocast(device_type="cuda", enabled=(USE_AMP and DEVICE.type == "cuda")):
                out = model(inp)

            out_np = tensor_to_hwc01(out)
            gt_np = tensor_to_hwc01(gt)

            psnr = compute_psnr(gt_np, out_np, data_range=1.0)
            ssim = compute_ssim(gt_np, out_np, channel_axis=2, data_range=1.0, win_size=7)

            psnr_sum += psnr
            ssim_sum += ssim
            n += 1

    return psnr_sum / max(n, 1), ssim_sum / max(n, 1)


# ======================
# Main
# ======================
def main():
    torch.backends.cudnn.benchmark = True
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    model = RestormerVolterra().to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scaler = GradScaler(device="cuda")

    for epoch in range(EPOCHS):
        epoch_num = epoch + 1
        transform, size = get_transform(epoch_num)

        # ----------------------
        # Balanced train dataset
        # ----------------------
        train_dataset, counts = build_balanced_train_dataset(transform, epoch_num)
        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKERS,
            pin_memory=(PIN_MEMORY and DEVICE.type == "cuda"),
            drop_last=True,
        )

        print(f"\n[Epoch {epoch_num}/{EPOCHS}] Unified Balanced(2000/task) | input={size}x{size}")
        print(f"  - {counts}")

        # ----------------------
        # Train
        # ----------------------
        model.train()
        loss_sum, psnr_sum, ssim_sum, count = 0.0, 0.0, 0.0, 0

        loop = tqdm(train_loader, desc=f"Train [{epoch_num}/{EPOCHS}]", leave=False)
        t0 = time.time()

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

            # metric (현재 배치 기준)
            out_np = tensor_to_hwc01(out)
            gt_np = tensor_to_hwc01(gt)

            psnr = compute_psnr(gt_np, out_np, data_range=1.0)
            ssim = compute_ssim(gt_np, out_np, channel_axis=2, data_range=1.0, win_size=7)

            loss_sum += float(loss.item())
            psnr_sum += float(psnr)
            ssim_sum += float(ssim)
            count += 1

            loop.set_postfix(
                loss=f"{loss.item():.4f}",
                psnr=f"{psnr:.2f}",
                ssim=f"{ssim:.3f}"
            )

        dt = time.time() - t0
        train_loss = loss_sum / max(count, 1)
        train_psnr = psnr_sum / max(count, 1)
        train_ssim = ssim_sum / max(count, 1)

        print(f"[Train] loss={train_loss:.6f} | psnr={train_psnr:.2f} | ssim={train_ssim:.4f} | time={dt/60:.1f}m")

        # ----------------------
        # Validation (task-wise)
        # ----------------------
        val_transform = transforms.Compose([transforms.Resize((size, size)), transforms.ToTensor()])

        val_r100h = DataLoader(
            Rain100HDataset(root_dir=RAIN100H_TEST, transform=val_transform),
            batch_size=1, shuffle=False, num_workers=NUM_WORKERS,
            pin_memory=(PIN_MEMORY and DEVICE.type == "cuda")
        )
        val_r100l = DataLoader(
            Rain100LDataset(root_dir=RAIN100L_TEST, transform=val_transform),
            batch_size=1, shuffle=False, num_workers=NUM_WORKERS,
            pin_memory=(PIN_MEMORY and DEVICE.type == "cuda")
        )
        val_gopro = DataLoader(
            GoProDataset(GOPRO_CSV_TEST, transform=val_transform),
            batch_size=1, shuffle=False, num_workers=NUM_WORKERS,
            pin_memory=(PIN_MEMORY and DEVICE.type == "cuda")
        )
        val_reside = DataLoader(
            RESIDEDataset(root_dir=RESIDE_ROOT, split="test", transform=val_transform, strict=True),
            batch_size=1, shuffle=False, num_workers=NUM_WORKERS,
            pin_memory=(PIN_MEMORY and DEVICE.type == "cuda")
        )
        val_csd = DataLoader(
            CSDDataset(root_dir=CSD_ROOT, split="test", transform=val_transform, strict=True),
            batch_size=1, shuffle=False, num_workers=NUM_WORKERS,
            pin_memory=(PIN_MEMORY and DEVICE.type == "cuda")
        )

        psnr_h, ssim_h = eval_loader(model, val_r100h, "Rain100H")
        psnr_l, ssim_l = eval_loader(model, val_r100l, "Rain100L")
        psnr_g, ssim_g = eval_loader(model, val_gopro, "GoPro")
        psnr_z, ssim_z = eval_loader(model, val_reside, "RESIDE")
        psnr_s, ssim_s = eval_loader(model, val_csd, "CSD")

        # Rain을 하나의 task로 보고 평균
        psnr_rain = (psnr_h + psnr_l) / 2.0
        ssim_rain = (ssim_h + ssim_l) / 2.0

        # 4-task 평균: Rain(avg), GoPro, RESIDE, CSD
        psnr_avg = (psnr_rain + psnr_g + psnr_z + psnr_s) / 4.0
        ssim_avg = (ssim_rain + ssim_g + ssim_z + ssim_s) / 4.0

        print("\n[Val] Task-wise results")
        print(f"  Rain100H : PSNR {psnr_h:.2f} | SSIM {ssim_h:.4f}")
        print(f"  Rain100L : PSNR {psnr_l:.2f} | SSIM {ssim_l:.4f}")
        print(f"  Rain(avg): PSNR {psnr_rain:.2f} | SSIM {ssim_rain:.4f}")
        print(f"  GoPro    : PSNR {psnr_g:.2f} | SSIM {ssim_g:.4f}")
        print(f"  RESIDE   : PSNR {psnr_z:.2f} | SSIM {ssim_z:.4f}")
        print(f"  CSD      : PSNR {psnr_s:.2f} | SSIM {ssim_s:.4f}")
        print(f"  AVG(4)   : PSNR {psnr_avg:.2f} | SSIM {ssim_avg:.4f}")

        # ----------------------
        # Save checkpoint (validation 평균으로 파일명)
        # ----------------------
        ckpt_name = f"epoch_{epoch_num:03d}_ssim{ssim_avg:.4f}_psnr{psnr_avg:.2f}.pth"
        ckpt_path = os.path.join(SAVE_DIR, ckpt_name)
        torch.save(model.state_dict(), ckpt_path)
        print(f"[Save] {ckpt_path}")


if __name__ == "__main__":
    main()

"""
epoch 1
  Rain100H : PSNR 22.26 | SSIM 0.6907
  Rain100L : PSNR 23.71 | SSIM 0.8072
  Rain(avg): PSNR 22.98 | SSIM 0.7489
  GoPro    : PSNR 25.61 | SSIM 0.8971
  RESIDE   : PSNR 21.87 | SSIM 0.8075
  CSD      : PSNR 22.74 | SSIM 0.7573
  AVG(4)   : PSNR 23.30 | SSIM 0.8027

Rain100H : PSNR 23.12 | SSIM 0.7432
  Rain100L : PSNR 24.33 | SSIM 0.8517
  Rain(avg): PSNR 23.72 | SSIM 0.7974
  GoPro    : PSNR 26.95 | SSIM 0.9297
  RESIDE   : PSNR 23.19 | SSIM 0.8809
  CSD      : PSNR 24.11 | SSIM 0.8238
  AVG(4)   : PSNR 24.49 | SSIM 0.8580

    Rain100H : PSNR 28.44 | SSIM 0.8708
  Rain100L : PSNR 33.33 | SSIM 0.9543
  Rain(avg): PSNR 30.89 | SSIM 0.9125
  GoPro    : PSNR 31.87 | SSIM 0.9615
  RESIDE   : PSNR 28.28 | SSIM 0.9499
  CSD      : PSNR 30.29 | SSIM 0.9310


  [Val] Task-wise results
  Rain100H : PSNR 29.49 | SSIM 0.8807
  Rain100L : PSNR 34.52 | SSIM 0.9623
  Rain(avg): PSNR 32.01 | SSIM 0.9215
  GoPro    : PSNR 30.68 | SSIM 0.9415
  RESIDE   : PSNR 29.87 | SSIM 0.9581
  CSD      : PSNR 31.74 | SSIM 0.9425

  [Val] Task-wise results
  Rain100H : PSNR 29.31 | SSIM 0.8846
  Rain100L : PSNR 34.97 | SSIM 0.9633
  Rain(avg): PSNR 32.14 | SSIM 0.9239
  GoPro    : PSNR 31.40 | SSIM 0.9452
  RESIDE   : PSNR 29.62 | SSIM 0.9617
  CSD      : PSNR 31.77 | SSIM 0.9442
  AVG(4)   : PSNR 31.23 | SSIM 0.9438

    Rain100H : PSNR 29.62 | SSIM 0.8859
  Rain100L : PSNR 34.56 | SSIM 0.9636
  Rain(avg): PSNR 32.09 | SSIM 0.9247
  GoPro    : PSNR 29.88 | SSIM 0.9368
  RESIDE   : PSNR 30.37 | SSIM 0.9622
  CSD      : PSNR 31.90 | SSIM 0.9437
  AVG(4)   : PSNR 31.06 | SSIM 0.9418

    Rain100H : PSNR 30.03 | SSIM 0.8934
  Rain100L : PSNR 34.54 | SSIM 0.9651
  Rain(avg): PSNR 32.29 | SSIM 0.9292
  GoPro    : PSNR 30.19 | SSIM 0.9285
  RESIDE   : PSNR 29.72 | SSIM 0.9573
  CSD      : PSNR 31.48 | SSIM 0.9499
  AVG(4)   : PSNR 30.92 | SSIM 0.9412
"""