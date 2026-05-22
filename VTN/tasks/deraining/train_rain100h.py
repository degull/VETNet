""" import os
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
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim

from models.restormer_volterra import RestormerVolterra
from re_dataset.rain100h_dataset import Rain100HDataset


# ---------------- Config ----------------
BATCH_SIZE = 2
EPOCHS     = 100
LR         = 2e-4
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRAIN_DIR  = r"E:/restormer+volterra/data/rain100H/train"
TEST_DIR   = r"E:/restormer+volterra/data/rain100H/test"   # ✅ 평가용 test set

# ✅ 저장 경로
SAVE_DIR   = r"E:/restormer+volterra/Restormer + Volterra/tasks/deraining/checkpoint_rain100hs"
os.makedirs(SAVE_DIR, exist_ok=True)

resize_schedule = {0: 128, 30: 192, 60: 256}


# ---------------- Transform ----------------
def get_transform(epoch: int):
    size = max(v for k, v in resize_schedule.items() if epoch >= k)
    return transforms.Compose([
        transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor()
    ])


# ---------------- Evaluation ----------------
def evaluate(model, test_dir, transform):
    model.eval()
    input_dir = os.path.join(test_dir, "rain")
    target_dir = os.path.join(test_dir, "norain")

    input_files = sorted(os.listdir(input_dir))
    total_psnr = total_ssim = count = 0

    with torch.no_grad():
        for fname in input_files:
            input_path = os.path.join(input_dir, fname)
            target_path = os.path.join(target_dir, fname)
            if not os.path.isfile(target_path):
                continue

            input_img = transform(Image.open(input_path).convert("RGB"))
            target_img = transform(Image.open(target_path).convert("RGB"))

            input_img  = input_img.unsqueeze(0).to(DEVICE)
            target_img = target_img.unsqueeze(0).to(DEVICE)

            output = model(input_img)

            out_np = output[0].detach().cpu().numpy().transpose(1, 2, 0)
            tgt_np = target_img[0].cpu().numpy().transpose(1, 2, 0)

            psnr = compute_psnr(tgt_np, out_np, data_range=1.0)
            ssim = compute_ssim(tgt_np, out_np, data_range=1.0, channel_axis=2, win_size=7)

            total_psnr += psnr
            total_ssim += ssim
            count += 1

    avg_psnr = total_psnr / count
    avg_ssim = total_ssim / count
    return avg_psnr, avg_ssim


# ---------------- Main ----------------
def main():
    model     = RestormerVolterra().to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scaler    = GradScaler()

    print(f"\n[INFO] Training Rain100H (Save checkpoints with SSIM+PSNR)\n")

    for epoch in range(EPOCHS):
        transform = get_transform(epoch)

        train_ds = Rain100HDataset(root_dir=TRAIN_DIR, transform=transform)
        train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=True)

        print(f"[Epoch {epoch+1:3d}] Input size: {transform.transforms[0].size} | Train Samples: {len(train_ds)}")

        model.train()
        epoch_loss = 0.0

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
            loop.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = epoch_loss / len(train_dl)
        print(f"[Epoch {epoch+1:3d}] Train Loss: {avg_loss:.6f}")

        # --- Evaluate on test set ---
        test_psnr, test_ssim = evaluate(model, TEST_DIR, transform)
        print(f"✅ [Epoch {epoch+1:3d}] Test PSNR: {test_psnr:.2f} | SSIM: {test_ssim:.4f}")

        # --- Save checkpoint with PSNR / SSIM in filename ---
        save_name = f"epoch_{epoch+1}_ssim{test_ssim:.4f}_psnr{test_psnr:.2f}.pth"
        torch.save({"model": model.state_dict()}, os.path.join(SAVE_DIR, save_name))


if __name__ == "__main__":
    main()
 """

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
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim

from models.restormer_volterra import RestormerVolterra
from re_dataset.rain100h_dataset import Rain100HDataset


# ---------------- Config ----------------
BATCH_SIZE = 2
EPOCHS     = 100
LR         = 2e-4
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRAIN_DIR  = r"E:/restormer+volterra/data/rain100H/train"
TEST_DIR   = r"E:/restormer+volterra/data/rain100H/test"   # ✅ 평가용 test set

# ✅ 저장 경로
SAVE_DIR   = r"E:/restormer+volterra/Restormer + Volterra/tasks/deraining/checkpoint_rain100hs"
os.makedirs(SAVE_DIR, exist_ok=True)

# ✅ 이어서 학습할 checkpoint
RESUME_CKPT  = r"E:/restormer+volterra/Restormer + Volterra/tasks/deraining/checkpoint_rain100hs/epoch_7_ssim0.8286_psnr25.37.pth"
START_EPOCH  = 7   # 이어서 시작할 epoch 번호 (파일명 기준)

resize_schedule = {0: 128, 30: 192, 60: 256}


# ---------------- Transform ----------------
def get_transform(epoch: int):
    size = max(v for k, v in resize_schedule.items() if epoch >= k)
    return transforms.Compose([
        transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor()
    ])


# ---------------- Evaluation ----------------
def evaluate(model, test_dir, transform):
    model.eval()
    input_dir = os.path.join(test_dir, "rain")
    target_dir = os.path.join(test_dir, "norain")

    input_files = sorted(os.listdir(input_dir))
    total_psnr = total_ssim = count = 0

    with torch.no_grad():
        for fname in input_files:
            input_path = os.path.join(input_dir, fname)
            target_path = os.path.join(target_dir, fname)
            if not os.path.isfile(target_path):
                continue

            input_img = transform(Image.open(input_path).convert("RGB"))
            target_img = transform(Image.open(target_path).convert("RGB"))

            input_img  = input_img.unsqueeze(0).to(DEVICE)
            target_img = target_img.unsqueeze(0).to(DEVICE)

            output = model(input_img)

            out_np = output[0].detach().cpu().numpy().transpose(1, 2, 0)
            tgt_np = target_img[0].cpu().numpy().transpose(1, 2, 0)

            psnr = compute_psnr(tgt_np, out_np, data_range=1.0)
            ssim = compute_ssim(tgt_np, out_np, data_range=1.0, channel_axis=2, win_size=7)

            total_psnr += psnr
            total_ssim += ssim
            count += 1

    avg_psnr = total_psnr / count
    avg_ssim = total_ssim / count
    return avg_psnr, avg_ssim


# ---------------- Main ----------------
def main():
    model     = RestormerVolterra().to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scaler    = GradScaler()

    start_epoch = 0

    # ✅ 이어서 학습 로드
    if os.path.exists(RESUME_CKPT):
        print(f"[INFO] Resuming training from: {RESUME_CKPT}")
        checkpoint = torch.load(RESUME_CKPT, map_location=DEVICE)

        if "model" in checkpoint:
            model.load_state_dict(checkpoint["model"])
        else:
            model.load_state_dict(checkpoint)

        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        if "scaler" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler"])

        start_epoch = START_EPOCH
        print(f"[INFO] Resumed from epoch {start_epoch}")

    print(f"\n[INFO] Training Rain100H (Save checkpoints with SSIM+PSNR)\n")

    for epoch in range(start_epoch, EPOCHS):
        transform = get_transform(epoch)

        train_ds = Rain100HDataset(root_dir=TRAIN_DIR, transform=transform)
        train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=True)

        print(f"[Epoch {epoch+1:3d}] Input size: {transform.transforms[0].size} | Train Samples: {len(train_ds)}")

        model.train()
        epoch_loss = 0.0

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
            loop.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = epoch_loss / len(train_dl)
        print(f"[Epoch {epoch+1:3d}] Train Loss: {avg_loss:.6f}")

        # --- Evaluate on test set ---
        test_psnr, test_ssim = evaluate(model, TEST_DIR, transform)
        print(f"✅ [Epoch {epoch+1:3d}] Test PSNR: {test_psnr:.2f} | SSIM: {test_ssim:.4f}")

        # --- Save checkpoint with PSNR / SSIM in filename ---
        save_name = f"epoch_{epoch+1}_ssim{test_ssim:.4f}_psnr{test_psnr:.2f}.pth"
        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch+1,
            "psnr": test_psnr,
            "ssim": test_ssim
        }, os.path.join(SAVE_DIR, save_name))


if __name__ == "__main__":
    main()
