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
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm
from PIL import Image
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim
from torch.amp import autocast, GradScaler

from models.restormer_volterra import RestormerVolterra


class CSDDataset(Dataset):
    def __init__(self, snow_dir, gt_dir, transform=None):
        self.snow_files = sorted(os.listdir(snow_dir))
        self.snow_dir = snow_dir
        self.gt_dir = gt_dir
        self.transform = transform or transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.snow_files)

    def __getitem__(self, idx):
        snow_path = os.path.join(self.snow_dir, self.snow_files[idx])
        gt_path = os.path.join(self.gt_dir, self.snow_files[idx])
        snow = Image.open(snow_path).convert("RGB")
        gt = Image.open(gt_path).convert("RGB")
        return self.transform(snow), self.transform(gt)


def get_transform(epoch: int):
    schedule = {0: 128, 30: 192, 60: 256}
    size = max(v for k, v in schedule.items() if epoch >= k)
    return transforms.Compose([
        transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor()
    ])


def main():
    # 경로 설정
    TRAIN_SNOW_DIR = r"E:/restormer+volterra/data/CSD/Train/Snow"
    TRAIN_GT_DIR   = r"E:/restormer+volterra/data/CSD/Train/Gt"
    SAVE_DIR       = r"E:/restormer+volterra/checkpoints/restormer_volterra_csd"
    os.makedirs(SAVE_DIR, exist_ok=True)

    DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCHSIZE = 2
    EPOCHS    = 100
    LR        = 2e-4

    model     = RestormerVolterra().to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scaler    = GradScaler()

    print("\n[INFO] Training Only on CSD Dataset\n")

    for epoch in range(EPOCHS):
        transform = get_transform(epoch)
        train_ds = CSDDataset(TRAIN_SNOW_DIR, TRAIN_GT_DIR, transform=transform)
        train_dl = DataLoader(train_ds, batch_size=BATCHSIZE, shuffle=True, num_workers=4, pin_memory=True)

        model.train()
        total_loss = total_psnr = total_ssim = 0.0

        loop = tqdm(train_dl, desc=f"[Epoch {epoch+1}/{EPOCHS}]")

        for snow, gt in loop:
            snow = snow.to(DEVICE)
            gt = gt.to(DEVICE)

            optimizer.zero_grad(set_to_none=True)

            with autocast(device_type='cuda'):
                output = model(snow)
                loss = criterion(output, gt)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

            gt_np  = gt[0].cpu().numpy().transpose(1, 2, 0)
            out_np = output[0].detach().cpu().numpy().transpose(1, 2, 0)

            psnr = compute_psnr(gt_np, out_np, data_range=1.0)
            ssim = compute_ssim(gt_np, out_np, data_range=1.0, channel_axis=2, win_size=7)

            total_psnr += psnr
            total_ssim += ssim

            loop.set_postfix(loss=f"{loss.item():.4f}", psnr=f"{psnr:.2f}", ssim=f"{ssim:.3f}")

        avg_loss = total_loss / len(train_dl)
        avg_psnr = total_psnr / len(train_dl)
        avg_ssim = total_ssim / len(train_dl)

        print(f"[Epoch {epoch+1:03d}] Loss: {avg_loss:.6f} | PSNR: {avg_psnr:.2f} | SSIM: {avg_ssim:.4f}")

        ckpt_name = f"epoch_{epoch+1}_ssim{avg_ssim:.4f}_psnr{avg_psnr:.2f}.pth"
        torch.save(model.state_dict(), os.path.join(SAVE_DIR, ckpt_name))


if __name__ == "__main__":
    main()
