# file: test_complex_distortion.py
import os
import sys
import random
from glob import glob

import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim
from tqdm import tqdm

# ✅ Python path 설정
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(current_dir, '..')))      # multiple_distortion 상위
sys.path.append(os.path.abspath(os.path.join(current_dir, '..', '..'))) # 프로젝트 루트

from models.restormer_volterra import RestormerVolterra
from pipeline import apply_random_distortions   # ✅ ARNIQA distortion pipeline


# ---------------- Config ----------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ All-in-One 학습된 checkpoint
CKPT_PATH = r"E:\restormer+volterra\checkpoints\#01_all_tasks_balanced_160\epoch_99_ssim0.9183_psnr32.73.pth"

# ✅ 여러 GT 폴더
CLEAN_DIRS = [
    r"E:/restormer+volterra/data/CSD/Test/Gt",
    r"E:/restormer+volterra/data/HIDE/GT",
    r"E:/restormer+volterra/data/rain100H/test/norain",
    r"E:/restormer+volterra/data/rain100L/test/norain",
    r"E:/restormer+volterra/data/SIDD/Data",
]

RESULT_DIR = r"E:/restormer+volterra/results/complex_distortion"
os.makedirs(RESULT_DIR, exist_ok=True)

IMG_EXT = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]

to_tensor = transforms.ToTensor()
to_pil = transforms.ToPILImage()


# ---------------- Evaluation ----------------
def evaluate(model, clean_dirs):
    # 여러 폴더에서 이미지 모으기
    clean_files = []
    for d in clean_dirs:
        files = [f for f in glob(os.path.join(d, "*")) if os.path.splitext(f)[1].lower() in IMG_EXT]
        clean_files.extend(files)

    if len(clean_files) == 0:
        print(f"[ERROR] No images found in {clean_dirs}")
        return

    total_psnr, total_ssim, count = 0, 0, 0

    for fpath in tqdm(clean_files, desc="Testing Complex Distortions"):
        # Load GT image
        gt = Image.open(fpath).convert("RGB")

        # Apply ARNIQA-style random complex distortion
        distorted = apply_random_distortions(gt)

        # Convert to tensor
        inp = to_tensor(distorted).unsqueeze(0).to(DEVICE)
        tgt = to_tensor(gt).unsqueeze(0).to(DEVICE)

        # Forward pass
        with torch.no_grad():
            output = model(inp)

        out_np = output[0].cpu().numpy().transpose(1, 2, 0)
        tgt_np = tgt[0].cpu().numpy().transpose(1, 2, 0)

        psnr = compute_psnr(tgt_np, out_np, data_range=1.0)
        ssim = compute_ssim(tgt_np, out_np, data_range=1.0, channel_axis=2)

        total_psnr += psnr
        total_ssim += ssim
        count += 1

        # Save sample result
        out_img = to_pil(output[0].cpu().clamp(0, 1))
        out_name = os.path.basename(fpath).replace(".", "_restored.")
        out_img.save(os.path.join(RESULT_DIR, out_name))

    avg_psnr = total_psnr / count
    avg_ssim = total_ssim / count

    print(f"\n✅ [Complex Distortion Testset] Images: {count}")
    print(f"   PSNR: {avg_psnr:.2f} | SSIM: {avg_ssim:.4f}")


# ---------------- Main ----------------
def main():
    model = RestormerVolterra().to(DEVICE)

    checkpoint = torch.load(CKPT_PATH, map_location=DEVICE)
    print(f"[INFO] Loaded checkpoint: {CKPT_PATH}")

    if isinstance(checkpoint, dict):
        if "model" in checkpoint:
            state_dict = checkpoint["model"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict)

    evaluate(model, CLEAN_DIRS)


if __name__ == "__main__":
    main()
