# E:\restormer+volterra\Restormer + Volterra\test.py
""" import os, sys

CUR_DIR = os.path.dirname(os.path.abspath(__file__))                  # .../Restormer + Volterra
ROOT_DIR = os.path.dirname(CUR_DIR)                                  # .../restormer+volterra

for p in [CUR_DIR, ROOT_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from models.restormer_volterra import RestormerVolterra
from re_dataset.rain100l_dataset import Rain100LDataset
from re_dataset.rain100h_dataset import Rain100HDataset
from re_dataset.gopro_dataset import GoProDataset
from re_dataset.sidd_dataset import SIDD_Dataset

import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim



# ======================
# 설정
# ======================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

CHECKPOINT_PATH = r"E:\restormer+volterra\checkpoints\restormer_volterra_sidd\epoch_100.pth"
RAIN100L_DIR = r"E:\restormer+volterra\data\SIDD\Data"

# ======================
# 모델 로드
# ======================
model = RestormerVolterra().to(DEVICE)
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
model.eval()

# ======================
# 데이터셋
# ======================
dataset = SIDD_Dataset(root_dir=RAIN100L_DIR, transform=None)
loader = DataLoader(dataset, batch_size=1, shuffle=False)

# ======================
# 평가
# ======================
psnr_total, ssim_total = 0.0, 0.0

with torch.no_grad():
    for rainy, gt in tqdm(loader, desc="Evaluating Rain100L"):
        rainy = rainy.to(DEVICE)
        gt = gt.to(DEVICE)

        restored = model(rainy)

        # Tensor → numpy (HWC)
        restored = restored.squeeze(0).clamp(0, 1).cpu().numpy().transpose(1, 2, 0)
        gt = gt.squeeze(0).clamp(0, 1).cpu().numpy().transpose(1, 2, 0)

        psnr = compute_psnr(gt, restored, data_range=1.0)
        ssim = compute_ssim(gt, restored, data_range=1.0, channel_axis=2)

        psnr_total += psnr
        ssim_total += ssim

num_images = len(loader)

print("\n==============================")
print(f"📊 Rain100L Test Results")
print(f"✅ PSNR : {psnr_total / num_images:.2f} dB")
print(f"✅ SSIM : {ssim_total / num_images:.4f}")
print("==============================")

 """
# sidd 전용
# E:\restormer+volterra\Restormer + Volterra\test_sidd.py
import os, sys
import csv

import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim

# ----- path setup (models: current dir, re_dataset: repo root) -----
CUR_DIR = os.path.dirname(os.path.abspath(__file__))        # .../Restormer + Volterra
ROOT_DIR = os.path.dirname(CUR_DIR)                         # .../restormer+volterra
for p in [CUR_DIR, ROOT_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from models.restormer_volterra import RestormerVolterra


# ======================
# SIDD CSV Dataset
# ======================
class SIDDCSVDataset(Dataset):
    """
    CSV columns: dist_img, ref_img (or noisy, gt)
    Supports auto path remap when CSV contains old absolute paths.
    """
    def __init__(self, root_dir: str, csv_path: str):
        self.root_dir = root_dir
        self.pairs = []

        # ✅ 여기 두 줄만 네 상황에 맞춰서 고정
        OLD_PREFIX = "C:/Users/IIPL02/Desktop/MRVNet2D/dataset/SIDD_Small_sRGB_Only"
        NEW_PREFIX = root_dir.replace("\\", "/")  # e.g., E:/restormer+volterra/data/SIDD

        with open(csv_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) < 2:
                    continue

                a, b = row[0].strip(), row[1].strip()
                if not a or not b:
                    continue

                # ✅ 헤더 스킵 (dist_img,ref_img)
                if a.lower() in ["dist_img", "noisy", "input"] and b.lower() in ["ref_img", "gt", "target"]:
                    continue

                # 1) absolute or relative 처리
                noisy_path = a if os.path.isabs(a) else os.path.join(root_dir, a)
                gt_path    = b if os.path.isabs(b) else os.path.join(root_dir, b)

                # 2) ✅ 존재 안 하면 prefix remap 시도
                if not os.path.exists(noisy_path) and noisy_path.replace("\\", "/").startswith(OLD_PREFIX):
                    noisy_path = noisy_path.replace("\\", "/").replace(OLD_PREFIX, NEW_PREFIX)
                if not os.path.exists(gt_path) and gt_path.replace("\\", "/").startswith(OLD_PREFIX):
                    gt_path = gt_path.replace("\\", "/").replace(OLD_PREFIX, NEW_PREFIX)

                # 3) 다시 체크
                if os.path.exists(noisy_path) and os.path.exists(gt_path):
                    self.pairs.append((noisy_path, gt_path))

        if len(self.pairs) == 0:
            raise RuntimeError(
                f"No valid pairs found.\nroot_dir={root_dir}\ncsv={csv_path}\n"
                f"CSV paths likely point to a different machine/location.\n"
                f"Tried remap:\n  {OLD_PREFIX}\n-> {NEW_PREFIX}\n"
            )

    def __len__(self):
        return len(self.pairs)

    def _load_img(self, path: str) -> torch.Tensor:
        img = Image.open(path).convert("RGB")
        arr = np.asarray(img).astype(np.float32) / 255.0
        ten = torch.from_numpy(arr).permute(2, 0, 1)
        return ten

    def __getitem__(self, idx):
        noisy_path, gt_path = self.pairs[idx]
        noisy = self._load_img(noisy_path)
        gt = self._load_img(gt_path)
        return noisy, gt



# ======================
# 설정
# ======================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CHECKPOINT_PATH = r"E:\restormer+volterra\checkpoints\restormer_volterra_sidd\epoch_100.pth"

SIDD_ROOT = r"E:\restormer+volterra\data\SIDD"                 # CSV 기준 루트
SIDD_TEST_CSV = os.path.join(SIDD_ROOT, "sidd_test_pairs.csv") # ✅ test split

# ======================
# 모델 로드
# ======================
model = RestormerVolterra().to(DEVICE)
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
model.eval()

# ======================
# 데이터셋/로더
# ======================
dataset = SIDDCSVDataset(root_dir=SIDD_ROOT, csv_path=SIDD_TEST_CSV)
loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

# ======================
# 평가
# ======================
psnr_total, ssim_total = 0.0, 0.0

with torch.no_grad():
    for noisy, gt in tqdm(loader, desc="Evaluating SIDD (test CSV)"):
        noisy = noisy.to(DEVICE)
        gt = gt.to(DEVICE)

        restored = model(noisy)

        restored = restored.squeeze(0).clamp(0, 1).cpu().numpy().transpose(1, 2, 0)
        gt_np    = gt.squeeze(0).clamp(0, 1).cpu().numpy().transpose(1, 2, 0)

        psnr = compute_psnr(gt_np, restored, data_range=1.0)
        ssim = compute_ssim(gt_np, restored, data_range=1.0, channel_axis=2)

        psnr_total += psnr
        ssim_total += ssim

num_images = len(loader)

print("\n==============================")
print("📊 SIDD Test Results (sidd_test_pairs.csv)")
print(f"✅ PSNR : {psnr_total / num_images:.2f} dB")
print(f"✅ SSIM : {ssim_total / num_images:.4f}")
print("==============================")
