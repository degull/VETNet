import os

import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim
from models.restormer_volterra import RestormerVolterra



DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_PATH = r"E:/restormer+volterra/checkpoints/restormer_volterra_rain100h_joint/epoch_100.pth"

TEST_DIRS = {
    "Test100":  r"E:/restormer+volterra/data/Test100",
    "Test1200": r"E:/restormer+volterra/data/Test1200",
    "Test2800": r"E:/restormer+volterra/data/Test2800",
}

transform = transforms.Compose([
    transforms.ToTensor()
])


def evaluate_test_folder(model, input_dir, name):
    model.eval()
    input_files = sorted([f for f in os.listdir(input_dir) if f.endswith(('.png', '.jpg'))])

    total_psnr = total_ssim = count = 0

    with torch.no_grad():
        for fname in input_files:
            input_path = os.path.join(input_dir, fname)
            input_img = transform(Image.open(input_path).convert("RGB"))
            input_img = input_img.unsqueeze(0).to(DEVICE)

            output = model(input_img)

            out_np = output[0].detach().cpu().numpy().transpose(1, 2, 0)
            tgt_np = np.array(Image.open(input_path).convert("RGB")).astype(np.float32) / 255.0  # Input이 GT인 경우

            psnr = compute_psnr(tgt_np, out_np, data_range=1.0)
            ssim = compute_ssim(tgt_np, out_np, data_range=1.0, channel_axis=2, win_size=7)

            total_psnr += psnr
            total_ssim += ssim
            count += 1

    avg_psnr = total_psnr / count
    avg_ssim = total_ssim / count
    print(f"✅ {name}  PSNR: {avg_psnr:.2f} | SSIM: {avg_ssim:.4f}")


def main():
    model = RestormerVolterra().to(DEVICE)
    checkpoint = torch.load(CHECKPOINT_PATH)
    model.load_state_dict(checkpoint)
    print(f"[INFO] Checkpoint 로드 완료: {CHECKPOINT_PATH}")

    for name, path in TEST_DIRS.items():
        evaluate_test_folder(model, path, name)


if __name__ == "__main__":
    main()
