import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset, Subset
from torchvision import transforms
from tqdm import tqdm
from torch.amp import autocast, GradScaler
from PIL import Image
import pandas as pd
import numpy as np
import random
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim

# ✅ 모델 import 경로 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from models.restormer_volterra import RestormerVolterra


# ------------------ Dataset 정의 ------------------
class PairedFolderDataset(torch.utils.data.Dataset):
    def __init__(self, input_dir, target_dir, transform):
        valid_ext = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
        self.input_paths = sorted([
            os.path.join(input_dir, f)
            for f in os.listdir(input_dir)
            if f.lower().endswith(valid_ext)
        ])
        self.target_paths = []
        for inp_path in self.input_paths:
            fname = os.path.basename(inp_path)
            found = False
            for ext in valid_ext:
                gt_path = os.path.join(target_dir, os.path.splitext(fname)[0] + ext)
                if os.path.exists(gt_path):
                    self.target_paths.append(gt_path)
                    found = True
                    break
            if not found:
                print(f"[⚠️] GT not found for {fname}")
                self.input_paths.remove(inp_path)

        assert len(self.input_paths) == len(self.target_paths), "Input/target mismatch"
        self.transform = transform

    def __len__(self):
        return len(self.input_paths)

    def __getitem__(self, idx):
        inp = Image.open(self.input_paths[idx]).convert("RGB")
        tgt = Image.open(self.target_paths[idx]).convert("RGB")
        return self.transform(inp), self.transform(tgt)


class PairedCSVDataset(torch.utils.data.Dataset):
    def __init__(self, csv_path, transform):
        df = pd.read_csv(csv_path)
        self.paths = df.values.tolist()
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        inp_path, tgt_path = self.paths[idx]
        inp = Image.open(inp_path).convert("RGB")
        tgt = Image.open(tgt_path).convert("RGB")
        return self.transform(inp), self.transform(tgt)


# ------------------ 설정 ------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_DIR = r"E:/restormer+volterra/checkpoints/#01_all_tasks_balanced_160"
os.makedirs(SAVE_DIR, exist_ok=True)

START_EPOCH = 1     # ✅ 처음부터 학습
EPOCHS = 100
BATCH_SIZE = 2
LR = 2e-4
MAX_SAMPLES = 160  # 각 데이터셋에서 사용할 샘플 수


# ------------------ Transform ------------------
resize_schedule = {0: 128, 30: 192, 60: 256}
def get_transform(epoch):
    size = max(v for k, v in resize_schedule.items() if epoch >= k)
    return transforms.Compose([
        transforms.Resize((size, size), interpolation=Image.BICUBIC),
        transforms.ToTensor()
    ])


# ------------------ Dataset 샘플링 ------------------
def subset_dataset(dataset, max_samples):
    total = len(dataset)
    if total <= max_samples:
        return dataset
    indices = random.sample(range(total), max_samples)
    return Subset(dataset, indices)


# ------------------ Dataset 로딩 ------------------
def get_all_datasets(transform):
    print("\n📦 Preparing balanced datasets...")

    datasets = []

    rain100h = subset_dataset(PairedFolderDataset(
        r"E:/restormer+volterra/data/rain100H/train/rain",
        r"E:/restormer+volterra/data/rain100H/train/norain",
        transform
    ), MAX_SAMPLES)
    print(f"Rain100H      : {len(rain100h)} samples")
    datasets.append(rain100h)

    rain100l = subset_dataset(PairedFolderDataset(
        r"E:/restormer+volterra/data/rain100L/train/rain",
        r"E:/restormer+volterra/data/rain100L/train/norain",
        transform
    ), MAX_SAMPLES)
    print(f"Rain100L      : {len(rain100l)} samples")
    datasets.append(rain100l)

    gopro = subset_dataset(PairedCSVDataset(
        r"E:/restormer+volterra/data/GOPRO_Large/gopro_train_pairs.csv",
        transform
    ), MAX_SAMPLES)
    print(f"GoPro         : {len(gopro)} samples")
    datasets.append(gopro)

    sidd = subset_dataset(PairedCSVDataset(
        r"E:/restormer+volterra/data/SIDD/sidd_pairs.csv",
        transform
    ), MAX_SAMPLES)
    print(f"SIDD          : {len(sidd)} samples")
    datasets.append(sidd)

    csd = subset_dataset(PairedFolderDataset(
        r"E:/restormer+volterra/data/CSD/Train/Snow",
        r"E:/restormer+volterra/data/CSD/Train/Gt",
        transform
    ), MAX_SAMPLES)
    print(f"CSD           : {len(csd)} samples")
    datasets.append(csd)

    total = sum(len(d) for d in datasets)
    print(f"📊 Total training samples (balanced): {total} images\n")

    return ConcatDataset(datasets)


# ------------------ 학습 루프 ------------------
def train_all_tasks():
    model = RestormerVolterra().to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LR)
    scaler = GradScaler()
    criterion = nn.L1Loss()

    print("\n🚀 Training from scratch...\n")

    for epoch in range(START_EPOCH, EPOCHS + 1):
        transform = get_transform(epoch)
        dataset = get_all_datasets(transform)
        train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                                  num_workers=4, pin_memory=True)

        model.train()
        total_loss = total_psnr = total_ssim = count = 0

        loop = tqdm(train_loader, desc=f"[Epoch {epoch}/{EPOCHS}]")
        for inputs, targets in loop:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

            with autocast(device_type='cuda'):
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

            # Metric 계산
            outputs = outputs.detach().clamp(0, 1).cpu().numpy()
            targets = targets.cpu().numpy()
            for out, gt in zip(outputs, targets):
                out = np.transpose(out, (1, 2, 0))
                gt = np.transpose(gt, (1, 2, 0))
                total_psnr += compute_psnr(gt, out, data_range=1.0)
                total_ssim += compute_ssim(gt, out, data_range=1.0, channel_axis=-1)
                count += 1

            loop.set_postfix(loss=loss.item())

        avg_psnr = total_psnr / count
        avg_ssim = total_ssim / count
        print(f"📣 Epoch {epoch:03d}: PSNR={avg_psnr:.2f}, SSIM={avg_ssim:.4f}")

        ckpt_name = f"epoch_{epoch}_ssim{avg_ssim:.4f}_psnr{avg_psnr:.2f}.pth"
        torch.save(model.state_dict(), os.path.join(SAVE_DIR, ckpt_name))


# ------------------ 실행 ------------------
if __name__ == "__main__":
    train_all_tasks()


# 이어서 학습
""" import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset, Subset
from torchvision import transforms
from tqdm import tqdm
from torch.amp import autocast, GradScaler
from PIL import Image
import pandas as pd
import numpy as np
import random
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim

# ✅ 모델 import 경로 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from models.restormer_volterra import RestormerVolterra


# ------------------ Dataset 정의 ------------------
class PairedFolderDataset(torch.utils.data.Dataset):
    def __init__(self, input_dir, target_dir, transform):
        valid_ext = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
        self.input_paths = sorted([
            os.path.join(input_dir, f)
            for f in os.listdir(input_dir)
            if f.lower().endswith(valid_ext)
        ])
        self.target_paths = []
        for inp_path in self.input_paths:
            fname = os.path.basename(inp_path)
            found = False
            for ext in valid_ext:
                gt_path = os.path.join(target_dir, os.path.splitext(fname)[0] + ext)
                if os.path.exists(gt_path):
                    self.target_paths.append(gt_path)
                    found = True
                    break
            if not found:
                print(f"[⚠️] GT not found for {fname}")
                self.input_paths.remove(inp_path)

        assert len(self.input_paths) == len(self.target_paths), "Input/target mismatch"
        self.transform = transform

    def __len__(self):
        return len(self.input_paths)

    def __getitem__(self, idx):
        inp = Image.open(self.input_paths[idx]).convert("RGB")
        tgt = Image.open(self.target_paths[idx]).convert("RGB")
        return self.transform(inp), self.transform(tgt)


class PairedCSVDataset(torch.utils.data.Dataset):
    def __init__(self, csv_path, transform):
        df = pd.read_csv(csv_path)
        self.paths = df.values.tolist()
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        inp_path, tgt_path = self.paths[idx]
        inp = Image.open(inp_path).convert("RGB")
        tgt = Image.open(tgt_path).convert("RGB")
        return self.transform(inp), self.transform(tgt)


# ------------------ 설정 ------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_DIR = r"E:/restormer+volterra/checkpoints/#01_all_tasks_balanced_160"
os.makedirs(SAVE_DIR, exist_ok=True)

# ✅ 이어서 학습
RESUME_PATH = os.path.join(
    SAVE_DIR,
    "epoch_100_ssim0.9177_psnr32.58.pth"
)

START_EPOCH = 101   # 이어서 시작할 epoch
EPOCHS = 200        # 최종 epoch
BATCH_SIZE = 2
LR = 2e-4
MAX_SAMPLES = 160   # 각 데이터셋에서 사용할 샘플 수


# ------------------ Transform ------------------
resize_schedule = {0: 128, 30: 192, 60: 256}
def get_transform(epoch):
    size = max(v for k, v in resize_schedule.items() if epoch >= k)
    return transforms.Compose([
        transforms.Resize((size, size), interpolation=Image.BICUBIC),
        transforms.ToTensor()
    ])


# ------------------ Dataset 샘플링 ------------------
def subset_dataset(dataset, max_samples):
    total = len(dataset)
    if total <= max_samples:
        return dataset
    indices = random.sample(range(total), max_samples)
    return Subset(dataset, indices)


# ------------------ Dataset 로딩 ------------------
def get_all_datasets(transform):
    print("\n📦 Preparing balanced datasets...")

    datasets = []

    rain100h = subset_dataset(PairedFolderDataset(
        r"E:/restormer+volterra/data/rain100H/train/rain",
        r"E:/restormer+volterra/data/rain100H/train/norain",
        transform
    ), MAX_SAMPLES)
    print(f"Rain100H      : {len(rain100h)} samples")
    datasets.append(rain100h)

    rain100l = subset_dataset(PairedFolderDataset(
        r"E:/restormer+volterra/data/rain100L/train/rain",
        r"E:/restormer+volterra/data/rain100L/train/norain",
        transform
    ), MAX_SAMPLES)
    print(f"Rain100L      : {len(rain100l)} samples")
    datasets.append(rain100l)

    gopro = subset_dataset(PairedCSVDataset(
        r"E:/restormer+volterra/data/GOPRO_Large/gopro_train_pairs.csv",
        transform
    ), MAX_SAMPLES)
    print(f"GoPro         : {len(gopro)} samples")
    datasets.append(gopro)

    sidd = subset_dataset(PairedCSVDataset(
        r"E:/restormer+volterra/data/SIDD/sidd_pairs.csv",
        transform
    ), MAX_SAMPLES)
    print(f"SIDD          : {len(sidd)} samples")
    datasets.append(sidd)

    csd = subset_dataset(PairedFolderDataset(
        r"E:/restormer+volterra/data/CSD/Train/Snow",
        r"E:/restormer+volterra/data/CSD/Train/Gt",
        transform
    ), MAX_SAMPLES)
    print(f"CSD           : {len(csd)} samples")
    datasets.append(csd)

    total = sum(len(d) for d in datasets)
    print(f"📊 Total training samples (balanced): {total} images\n")

    return ConcatDataset(datasets)


# ------------------ 학습 루프 ------------------
def train_all_tasks():
    model = RestormerVolterra().to(DEVICE)

    # ✅ 이전 체크포인트 불러오기
    if os.path.exists(RESUME_PATH):
        model.load_state_dict(torch.load(RESUME_PATH, map_location=DEVICE))
        print(f"✅ Resumed from: {RESUME_PATH}")
    else:
        print(f"[⚠️] Resume checkpoint not found: {RESUME_PATH}")
        return

    optimizer = optim.AdamW(model.parameters(), lr=LR)
    scaler = GradScaler()
    criterion = nn.L1Loss()

    print("\n🚀 Resuming Training...\n")

    for epoch in range(START_EPOCH, EPOCHS + 1):
        transform = get_transform(epoch)
        dataset = get_all_datasets(transform)
        train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                                  num_workers=4, pin_memory=True)

        model.train()
        total_loss = total_psnr = total_ssim = count = 0

        loop = tqdm(train_loader, desc=f"[Epoch {epoch}/{EPOCHS}]")
        for inputs, targets in loop:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

            with autocast(device_type='cuda'):
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

            # Metric 계산
            outputs = outputs.detach().clamp(0, 1).cpu().numpy()
            targets = targets.cpu().numpy()
            for out, gt in zip(outputs, targets):
                out = np.transpose(out, (1, 2, 0))
                gt = np.transpose(gt, (1, 2, 0))
                total_psnr += compute_psnr(gt, out, data_range=1.0)
                total_ssim += compute_ssim(gt, out, data_range=1.0, channel_axis=-1)
                count += 1

            loop.set_postfix(loss=loss.item())

        avg_psnr = total_psnr / count
        avg_ssim = total_ssim / count
        print(f"📣 Epoch {epoch:03d}: PSNR={avg_psnr:.2f}, SSIM={avg_ssim:.4f}")

        ckpt_name = f"epoch_{epoch}_ssim{avg_ssim:.4f}_psnr{avg_psnr:.2f}.pth"
        torch.save(model.state_dict(), os.path.join(SAVE_DIR, ckpt_name))


# ------------------ 실행 ------------------
if __name__ == "__main__":
    train_all_tasks()
 """