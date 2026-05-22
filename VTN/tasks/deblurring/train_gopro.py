import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))

# ✅ models 위치 (Restormer + Volterra 폴더)
sys.path.append(os.path.abspath(os.path.join(current_dir, '..', '..')))

# ✅ re_dataset 위치 (restormer+volterra 루트 폴더)
sys.path.append(os.path.abspath(os.path.join(current_dir, '..', '..', '..')))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from torch.amp import autocast, GradScaler
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim
from PIL import Image

from models.restormer_volterra import RestormerVolterra
from re_dataset.gopro_dataset import GoProDataset


# ───── 설정 ─────
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 2
EPOCHS     = 100
LR         = 2e-4

GOPRO_TRAIN_CSV = r"E:/restormer+volterra/data/GOPRO_Large/gopro_train_pairs.csv"
SAVE_DIR        = r"E:/restormer+volterra/checkpoints/restormer_volterra_gopro"
os.makedirs(SAVE_DIR, exist_ok=True)

resize_schedule = {0: 128, 30: 192, 60: 256}


# ───── Resize schedule ─────
def get_transform(epoch: int):
    size = max(v for k, v in resize_schedule.items() if epoch >= k)
    return transforms.Compose([
        transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor()
    ])


# ───── Main 학습 루프 ─────
def main():
    model     = RestormerVolterra().to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scaler    = GradScaler()

    print(f"\n[INFO] Training Only on GoPro Dataset (No Evaluation)\n")

    for epoch in range(EPOCHS):
        transform = get_transform(epoch)
        train_ds = GoProDataset(csv_path=GOPRO_TRAIN_CSV, transform=transform)
        train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

        print(f"[Epoch {epoch+1:3d}] Input size: {transform.transforms[0].size} | Train Samples: {len(train_ds)}")

        model.train()
        epoch_loss = tot_psnr = tot_ssim = 0.0
        count = 0

        loop = tqdm(train_dl, desc=f"Training Epoch {epoch+1}/{EPOCHS}", leave=False)

        for distorted, reference in loop:
            distorted = distorted.to(DEVICE)
            reference = reference.to(DEVICE)

            optimizer.zero_grad(set_to_none=True)

            with autocast(device_type='cuda'):
                output = model(distorted)
                loss = criterion(output, reference)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()

            # PSNR/SSIM 계산 (모니터링용)
            ref_np = reference[0].cpu().numpy().transpose(1, 2, 0)
            out_np = output[0].detach().cpu().numpy().transpose(1, 2, 0)
            psnr = compute_psnr(ref_np, out_np, data_range=1.0)
            ssim = compute_ssim(ref_np, out_np, data_range=1.0, channel_axis=2, win_size=7)

            tot_psnr += psnr
            tot_ssim += ssim
            count += 1

            loop.set_postfix(loss=f"{loss.item():.4f}", psnr=f"{psnr:.2f}", ssim=f"{ssim:.3f}")

        avg_loss = epoch_loss / len(train_dl)
        avg_psnr = tot_psnr / count
        avg_ssim = tot_ssim / count

        print(f"[Epoch {epoch+1:3d}] Loss: {avg_loss:.6f} | PSNR: {avg_psnr:.2f} | SSIM: {avg_ssim:.4f}")

        ckpt_name = f"epoch_{epoch+1}_ssim{avg_ssim:.4f}_psnr{avg_psnr:.2f}.pth"
        torch.save(model.state_dict(), os.path.join(SAVE_DIR, ckpt_name))


if __name__ == "__main__":
    main()



# ver2
""" 
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from torch.amp import autocast, GradScaler
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim
from PIL import Image

from models.restormer_volterra import RestormerVolterra
from re_dataset.gopro_dataset import GoProDataset


# ───── 설정 ─────
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 2
EPOCHS     = 100
LR         = 2e-4

GOPRO_TRAIN_CSV = r"E:/restormer+volterra/data/GOPRO_Large/gopro_train_pairs.csv"
SAVE_DIR        = r"E:/restormer+volterra/checkpoints/restormer_volterra_gopro"
os.makedirs(SAVE_DIR, exist_ok=True)

resize_schedule = {0: 128, 30: 192, 60: 256}


# ───── Epoch별 이미지 크기 조절 ─────
def get_transform(epoch: int):
    size = max(v for k, v in resize_schedule.items() if epoch >= k)
    return transforms.Compose([
        transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor()
    ])


# ───── Main 학습 루프 ─────
def main():
    model     = RestormerVolterra().to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scaler    = GradScaler()

    print(f"\n[INFO] Training Only on GoPro Dataset (No Evaluation)\n")

    for epoch in range(EPOCHS):
        transform = get_transform(epoch)
        train_ds = GoProDataset(csv_path=GOPRO_TRAIN_CSV, transform=transform)
        train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=True)

        print(f"[Epoch {epoch+1:3d}] Input size: {transform.transforms[0].size} | Train Samples: {len(train_ds)}")

        model.train()
        epoch_loss = tot_psnr = tot_ssim = 0.0
        count = 0

        loop = tqdm(train_dl, desc=f"Training Epoch {epoch+1}/{EPOCHS}", leave=False)

        for distorted, reference in loop:
            distorted = distorted.to(DEVICE)
            reference = reference.to(DEVICE)

            optimizer.zero_grad(set_to_none=True)

            with autocast(device_type='cuda'):
                output = model(distorted)
                loss = criterion(output, reference)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()

            ref_np = reference[0].cpu().numpy().transpose(1, 2, 0)
            out_np = output[0].detach().cpu().numpy().transpose(1, 2, 0)
            psnr = compute_psnr(ref_np, out_np, data_range=1.0)
            ssim = compute_ssim(ref_np, out_np, data_range=1.0, channel_axis=2, win_size=7)

            tot_psnr += psnr
            tot_ssim += ssim
            count += 1

            loop.set_postfix(loss=f"{loss.item():.4f}", psnr=f"{psnr:.2f}", ssim=f"{ssim:.3f}")

        avg_loss = epoch_loss / len(train_dl)
        avg_psnr = tot_psnr / count
        avg_ssim = tot_ssim / count

        print(f"[Epoch {epoch+1:3d}] Loss: {avg_loss:.6f} | PSNR: {avg_psnr:.2f} | SSIM: {avg_ssim:.4f}")

        ckpt_name = f"epoch_{epoch+1}_ssim{avg_ssim:.4f}_psnr{avg_psnr:.2f}.pth"
        torch.save(model.state_dict(), os.path.join(SAVE_DIR, ckpt_name))


if __name__ == "__main__":
    main()
 """