import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))

# ✅ models 위치 (Restormer + Volterra 폴더)
sys.path.append(os.path.abspath(os.path.join(current_dir, '..', '..')))

import torch
from torchvision import transforms
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim
from models.restormer_volterra import RestormerVolterra


DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TEST_DIR = r"E:/restormer+volterra/data/rain100H/test"

CKPT_PATH = r"E:\restormer+volterra\Restormer + Volterra\tasks\deraining\checkpoint_rain100h\epoch_128_ssim0.9862_psnr41.30.pth"


def get_transform():
    return transforms.Compose([
        transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor()
    ])


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

            # 🔎 이미지별 결과 출력
            print(f"[{fname}] PSNR: {psnr:.2f} | SSIM: {ssim:.4f}")

    avg_psnr = total_psnr / count
    avg_ssim = total_ssim / count
    print(f"\n✅ [Rain100H Testset]  PSNR: {avg_psnr:.2f} | SSIM: {avg_ssim:.4f}")


def main():
    model = RestormerVolterra().to(DEVICE)

    checkpoint = torch.load(CKPT_PATH, map_location=DEVICE)
    print(f"[INFO] Loaded checkpoint: {CKPT_PATH}")

    # ✅ state_dict만 로드
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

    transform = get_transform()
    evaluate(model, TEST_DIR, transform)


if __name__ == "__main__":
    main()
