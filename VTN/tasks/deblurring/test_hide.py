import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))

# ✅ models 위치 (Restormer + Volterra 폴더)
sys.path.append(os.path.abspath(os.path.join(current_dir, '..', '..')))

import torch
from torchvision import transforms
from PIL import Image
from models.restormer_volterra import RestormerVolterra
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

HIDE_DIR = r"E:/restormer+volterra/data/HIDE"
CKPT_PATH = r"E:/restormer+volterra/checkpoints/restormer_volterra_gopro/epoch_100.pth"

def get_transform():
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

def evaluate(model, input_dir, target_dir, name):
    model.eval()
    total_psnr = total_ssim = count = 0
    tfm = get_transform()
    files = sorted(os.listdir(input_dir))

    for fname in files:
        inp_path = os.path.join(input_dir, fname)
        tgt_path = os.path.join(target_dir, fname)
        if not os.path.exists(tgt_path):
            continue

        inp = tfm(Image.open(inp_path).convert("RGB")).unsqueeze(0).to(DEVICE)
        tgt = tfm(Image.open(tgt_path).convert("RGB")).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            out = model(inp)

        out_np = out[0].cpu().numpy().transpose(1, 2, 0)
        tgt_np = tgt[0].cpu().numpy().transpose(1, 2, 0)

        psnr = compute_psnr(tgt_np, out_np, data_range=1.0)
        ssim = compute_ssim(tgt_np, out_np, data_range=1.0, channel_axis=2, win_size=7)

        total_psnr += psnr
        total_ssim += ssim
        count += 1

    print(f"✅ [{name}] PSNR: {total_psnr / count:.2f} | SSIM: {total_ssim / count:.4f}")

def main():
    model = RestormerVolterra().to(DEVICE)
    model.load_state_dict(torch.load(CKPT_PATH))
    evaluate(model,
             input_dir=os.path.join(HIDE_DIR, "test"),
             target_dir=os.path.join(HIDE_DIR, "GT"),
             name="HIDE")

if __name__ == "__main__":
    main()
