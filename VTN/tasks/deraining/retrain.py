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
from re_dataset.rain100l_dataset import Rain100LDataset


# ---------------- Config ----------------
BATCH_SIZE = 2
EPOCHS     = 100
LR         = 2e-4
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ 여기서 H/L 데이터셋 선택
DATASET    = "Rain100H"   # "Rain100L" 로 바꾸면 L 학습
TRAIN_DIR  = rf"E:/restormer+volterra/data/{DATASET}/train"
TEST_DIR   = rf"E:/restormer+volterra/data/{DATASET}/test"

# ✅ 저장 경로 (고정)
SAVE_DIR   = r"E:/restormer+volterra/Restormer + Volterra/tasks/deraining/checkpoint"
os.makedirs(SAVE_DIR, exist_ok=True)

resize_schedule = {0: 128, 30: 192, 60: 256}


# ---------------- Transform ----------------
def get_train_transform(epoch: int):
    """훈련용 progressive resize"""
    size = max(v for k, v in resize_schedule.items() if epoch >= k)
    return transforms.Compose([
        transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor()
    ])


def get_test_transform():
    """평가용 (원본 해상도 유지)"""
    return transforms.ToTensor()


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

    if DATASET == "Rain100H":
        DatasetClass = Rain100HDataset
    elif DATASET == "Rain100L":
        DatasetClass = Rain100LDataset
    else:
        raise ValueError("DATASET must be 'Rain100H' or 'Rain100L'")

    print(f"\n[INFO] Training {DATASET} Only (Save with SSIM+PSNR)\n")

    for epoch in range(EPOCHS):
        # --- Train ---
        train_transform = get_train_transform(epoch)
        train_ds = DatasetClass(root_dir=TRAIN_DIR, transform=train_transform)
        train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=True)

        print(f"[Epoch {epoch+1:3d}] Input size: {train_transform.transforms[0].size} "
              f"| Train Samples: {len(train_ds)}")

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

        # --- Evaluate (원본 해상도) ---
        test_transform = get_test_transform()
        test_psnr, test_ssim = evaluate(model, TEST_DIR, test_transform)
        print(f"✅ [Epoch {epoch+1:3d}] Test  PSNR: {test_psnr:.2f} | Test  SSIM: {test_ssim:.4f}")

        # --- Save with PSNR / SSIM in filename ---
        save_name = f"{DATASET.lower()}_epoch{epoch+1}_ssim{test_ssim:.4f}_psnr{test_psnr:.2f}.pth"
        torch.save(model.state_dict(), os.path.join(SAVE_DIR, save_name))


if __name__ == "__main__":
    main()
