""" import os, sys
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
from tqdm import tqdm
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim

# ✅ 모델 불러오기 (Restormer+Volterra)
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(current_dir, '..', '..')))
from models.restormer_volterra import RestormerVolterra


# ---------------- Dataset ----------------
class SIDDPairDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.df = pd.read_csv(csv_file)
        self.transform = transform or transforms.Compose([
            transforms.Resize((256,256)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        noisy_path = self.df.iloc[idx, 0]
        gt_path    = self.df.iloc[idx, 1]
        noisy  = Image.open(noisy_path).convert("RGB")
        clean  = Image.open(gt_path).convert("RGB")
        return self.transform(noisy), self.transform(clean)


# ---------------- Evaluation ----------------
@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    total_psnr = total_ssim = 0.0
    n = 0

    for noisy, clean in tqdm(dataloader, desc="[Eval-SIDD]"):
        noisy, clean = noisy.to(device), clean.to(device)
        out = model(noisy).clamp(0,1)

        out_np   = out[0].cpu().numpy().transpose(1,2,0)
        clean_np = clean[0].cpu().numpy().transpose(1,2,0)

        psnr = compute_psnr(clean_np, out_np, data_range=1.0)
        ssim = compute_ssim(clean_np, out_np, channel_axis=2, data_range=1.0)

        total_psnr += psnr
        total_ssim += ssim
        n += 1

    avg_psnr = total_psnr / n
    avg_ssim = total_ssim / n
    print(f"\n📊 SIDD Test Results")
    print(f"Images: {n} | PSNR: {avg_psnr:.2f} dB | SSIM: {avg_ssim:.4f}")


# ---------------- Main ----------------
def main():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ✅ CSV 파일 경로
    CSV_PATH  = r"E:/restormer+volterra/data/SIDD/sidd_test_pairs.csv"
    CKPT_PATH = r"E:\restormer+volterra\checkpoints\restormer_volterra_rain100h\epoch_100.pth"

    # ✅ Dataset / DataLoader
    test_ds = SIDDPairDataset(CSV_PATH)
    test_dl = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

    # ✅ 모델 불러오기
    model = RestormerVolterra().to(DEVICE)
    state = torch.load(CKPT_PATH, map_location=DEVICE)
    model.load_state_dict(state, strict=False)
    print(f"✓ Loaded checkpoint: {CKPT_PATH}")

    # ✅ 평가
    evaluate(model, test_dl, DEVICE)


if __name__ == "__main__":
    main() """

# Deblurring
import os, sys
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim

# ✅ 모델 import
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(current_dir, '..', '..')))
from models.restormer_volterra import RestormerVolterra


# ---------------- Dataset ----------------
class HIDEFolderDataset(Dataset):
    def __init__(self, input_dir, target_dir, transform=None):
        self.input_paths = []
        for root, _, files in os.walk(input_dir):
            for f in files:
                if f.lower().endswith((".png", ".jpg", ".jpeg")):
                    self.input_paths.append(os.path.join(root, f))
        self.input_paths = sorted(self.input_paths)

        # GT dict: {stem → path}
        self.gt_dict = {
            os.path.splitext(f)[0]: os.path.join(target_dir, f)
            for f in os.listdir(target_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        }

        # 매칭된 페어
        self.pairs = []
        for inp in self.input_paths:
            stem = os.path.splitext(os.path.basename(inp))[0]
            if stem in self.gt_dict:
                self.pairs.append((inp, self.gt_dict[stem]))

        print(f"✓ Found {len(self.pairs)} valid pairs in {input_dir}")
        self.transform = transform or transforms.Compose([
            transforms.Resize((256,256)),
            transforms.ToTensor()
        ])

    def __len__(self): return len(self.pairs)

    def __getitem__(self, idx):
        inp, gt = self.pairs[idx]
        inp_img = Image.open(inp).convert("RGB")
        gt_img  = Image.open(gt).convert("RGB")
        return self.transform(inp_img), self.transform(gt_img)


# ---------------- Evaluation ----------------
@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    total_psnr = total_ssim = 0.0
    n = 0

    for x, y in tqdm(dataloader, desc="[Eval-HIDE]"):
        x, y = x.to(device), y.to(device)
        out = model(x).clamp(0,1)

        out_np = out[0].cpu().numpy().transpose(1,2,0)
        gt_np  = y[0].cpu().numpy().transpose(1,2,0)

        psnr = compute_psnr(gt_np, out_np, data_range=1.0)
        ssim = compute_ssim(gt_np, out_np, channel_axis=2, data_range=1.0)

        total_psnr += psnr
        total_ssim += ssim
        n += 1

    avg_psnr = total_psnr / n
    avg_ssim = total_ssim / n
    print(f"\n📊 HIDE Test Results")
    print(f"Images: {n} | PSNR: {avg_psnr:.2f} dB | SSIM: {avg_ssim:.4f}")


# ---------------- Main ----------------
def main():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    TEST_INPUT_DIR = r"E:/restormer+volterra/data/HIDE/test"
    GT_DIR         = r"E:/restormer+volterra/data/HIDE/GT"
    CKPT_PATH = r"E:\restormer+volterra\checkpoints\restormer_volterra_rain100h\epoch_100.pth"

    # ✅ Dataset
    test_ds = HIDEFolderDataset(TEST_INPUT_DIR, GT_DIR)
    test_dl = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

    # ✅ Model
    model = RestormerVolterra().to(DEVICE)
    state = torch.load(CKPT_PATH, map_location=DEVICE)
    model.load_state_dict(state, strict=False)
    print(f"✓ Loaded checkpoint: {CKPT_PATH}")

    # ✅ Eval
    evaluate(model, test_dl, DEVICE)


if __name__ == "__main__":
    main()