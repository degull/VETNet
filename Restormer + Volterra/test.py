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
# reside
# E:\restormer+volterra\Restormer + Volterra\test_reside.py
# RESIDE-6K (Dehazing) 테스트 스크립트
# - 최종 pth 로드
# - test split 전체 평균 PSNR/SSIM 출력
# - (옵션) 결과 이미지 저장: input|restored|gt PNG + 아래 PSNR/SSIM 텍스트

import os
import sys
import re
import csv

import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms

from torch.amp import autocast
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim

from PIL import Image, ImageDraw, ImageFont

# ----------------------
# Path setup (models: current dir, re_dataset: repo root)
# ----------------------
CUR_DIR = os.path.dirname(os.path.abspath(__file__))  # .../Restormer + Volterra
ROOT_DIR = os.path.dirname(CUR_DIR)                   # .../restormer+volterra
for p in [CUR_DIR, ROOT_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from models.restormer_volterra import RestormerVolterra
from re_dataset.reside_dataset import RESIDEDataset


# ======================
# Config
# ======================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ 최종 pth 지정 (여기만 바꾸면 됨)
CKPT_PATH = r"E:\restormer+volterra\checkpoints\restormer_volterra_reside\epoch_015_ssim0.9520_psnr27.97.pth"

RESIDE_ROOT = r"E:/restormer+volterra/data/RESIDE-6K"

# 평가 시 resize (train과 동일하게 맞추려면 256 권장)
EVAL_SIZE = 256

# AMP 사용
USE_AMP = True

# 결과 이미지 저장 여부
SAVE_RESULTS = True
NUM_SAVE = 20  # 앞에서 몇 장 저장할지
RESULTS_DIR = r"E:/restormer+volterra/results/reside_test_final"
os.makedirs(RESULTS_DIR, exist_ok=True)


# ======================
# Utils
# ======================
def tensor_to_hwc01(x: torch.Tensor) -> np.ndarray:
    if x.dim() == 4:
        x = x[0]
    x = x.detach().float().clamp(0, 1).cpu().numpy()
    x = np.transpose(x, (1, 2, 0))
    return x


def hwc01_to_pil(img: np.ndarray) -> Image.Image:
    img_u8 = (np.clip(img, 0, 1) * 255.0).astype(np.uint8)
    return Image.fromarray(img_u8, mode="RGB")


def render_triplet_with_text(inp_np, out_np, gt_np, psnr, ssim,
                             title_left="Input(Hazy)", title_mid="Restored", title_right="GT",
                             pad=12, text_h=54) -> Image.Image:
    inp_p = hwc01_to_pil(inp_np)
    out_p = hwc01_to_pil(out_np)
    gt_p = hwc01_to_pil(gt_np)

    w, h = inp_p.size
    canvas_w = w * 3 + pad * 4
    canvas_h = h + text_h + pad * 3
    canvas = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    try:
        font_b = ImageFont.truetype("arial.ttf", 20)
    except Exception:
        font_b = ImageFont.load_default()

    y_title = pad
    draw.text((pad + (w // 2) - 40, y_title), title_left, fill=(0, 0, 0), font=font_b)
    draw.text((pad * 2 + w + (w // 2) - 35, y_title), title_mid, fill=(0, 0, 0), font=font_b)
    draw.text((pad * 3 + w * 2 + (w // 2) - 10, y_title), title_right, fill=(0, 0, 0), font=font_b)

    y_img = pad * 2 + 18
    x1 = pad
    x2 = pad * 2 + w
    x3 = pad * 3 + w * 2
    canvas.paste(inp_p, (x1, y_img))
    canvas.paste(out_p, (x2, y_img))
    canvas.paste(gt_p, (x3, y_img))

    y_text = y_img + h + pad
    metric_text = f"PSNR: {psnr:.2f} dB    SSIM: {ssim:.4f}"
    draw.text((pad, y_text), metric_text, fill=(0, 0, 0), font=font_b)

    return canvas


# ======================
# Main
# ======================
def main():
    # transform (train과 맞추려면 Resize 유지)
    transform = transforms.Compose([
        transforms.Resize((EVAL_SIZE, EVAL_SIZE)),
        transforms.ToTensor()
    ])

    # dataset/loader
    test_set = RESIDEDataset(root_dir=RESIDE_ROOT, split="test", transform=transform, strict=True)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=4, pin_memory=(DEVICE.type == "cuda"))

    # model load
    model = RestormerVolterra().to(DEVICE)
    state = torch.load(CKPT_PATH, map_location=DEVICE)
    model.load_state_dict(state, strict=True)
    model.eval()

    psnr_total, ssim_total, n = 0.0, 0.0, 0
    saved = 0

    print("\n==============================")
    print("📊 RESIDE-6K Test (Final CKPT)")
    print(f"✅ ckpt : {CKPT_PATH}")
    print(f"✅ test : {len(test_set)} images")
    print(f"✅ eval resize : {EVAL_SIZE}x{EVAL_SIZE}")
    print(f"✅ amp : {USE_AMP}")
    print("==============================\n")

    with torch.no_grad():
        for hazy, gt in tqdm(test_loader, desc="Testing"):
            hazy = hazy.to(DEVICE, non_blocking=True)
            gt = gt.to(DEVICE, non_blocking=True)

            with autocast(device_type="cuda", enabled=(USE_AMP and DEVICE.type == "cuda")):
                out = model(hazy)

            hazy_np = tensor_to_hwc01(hazy)
            out_np = tensor_to_hwc01(out)
            gt_np = tensor_to_hwc01(gt)

            psnr = compute_psnr(gt_np, out_np, data_range=1.0)
            ssim = compute_ssim(gt_np, out_np, channel_axis=2, data_range=1.0, win_size=7)

            psnr_total += psnr
            ssim_total += ssim
            n += 1

            # 이미지별 metric 출력
            # print(f"[{n:04d}/{len(test_set):04d}] PSNR {psnr:.2f} | SSIM {ssim:.4f}")

            # 결과 저장
            if SAVE_RESULTS and saved < NUM_SAVE:
                vis = render_triplet_with_text(hazy_np, out_np, gt_np, psnr, ssim)
                vis.save(os.path.join(RESULTS_DIR, f"{saved+1:03d}_psnr{psnr:.2f}_ssim{ssim:.4f}.png"))
                saved += 1

    print("\n==============================")
    print("✅ Final Average Results")
    print(f"✅ PSNR : {psnr_total / max(n,1):.2f} dB")
    print(f"✅ SSIM : {ssim_total / max(n,1):.4f}")
    print("==============================")
    if SAVE_RESULTS:
        print(f"[Saved] {saved} triplet PNGs -> {RESULTS_DIR}")


if __name__ == "__main__":
    main()
