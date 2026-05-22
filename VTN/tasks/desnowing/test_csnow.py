import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))

# ✅ models 위치 (Restormer + Volterra 폴더)
sys.path.append(os.path.abspath(os.path.join(current_dir, '..', '..')))

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim

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


def evaluate(model, dataloader, device):
    model.eval()
    total_psnr = total_ssim = 0.0

    with torch.no_grad():
        for snow, gt in dataloader:
            snow = snow.to(device)
            gt   = gt.to(device)
            output = model(snow)

            gt_np  = gt[0].cpu().numpy().transpose(1, 2, 0)
            out_np = output[0].detach().cpu().numpy().transpose(1, 2, 0)

            psnr = compute_psnr(gt_np, out_np, data_range=1.0)
            ssim = compute_ssim(gt_np, out_np, data_range=1.0, channel_axis=2, win_size=7)

            total_psnr += psnr
            total_ssim += ssim

    avg_psnr = total_psnr / len(dataloader)
    avg_ssim = total_ssim / len(dataloader)

    print(f"\n📊 Test on CSD\nPSNR: {avg_psnr:.2f} dB | SSIM: {avg_ssim:.4f}")


def main():
    TEST_SNOW_DIR = r"E:/restormer+volterra/data/CSD/Test/Snow"
    TEST_GT_DIR   = r"E:/restormer+volterra/data/CSD/Test/Gt"
    CKPT_PATH     = r"E:\restormer+volterra\checkpoints\restormer_volterra_rain100l\epoch_91_ssim0.9538_psnr33.98.pth"
    # CKPT_PATH     = r"E:\restormer+volterra\checkpoints\restormer_volterra_rain100h\epoch_100.pth"

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = RestormerVolterra().to(DEVICE)
    model.load_state_dict(torch.load(CKPT_PATH))
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    test_ds = CSDDataset(TEST_SNOW_DIR, TEST_GT_DIR, transform=transform)
    test_dl = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

    evaluate(model, test_dl, DEVICE)


if __name__ == "__main__":
    main()
