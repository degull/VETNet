# Deraining
""" 
import os
import sys
from glob import glob
from natsort import natsorted

current_dir = os.path.dirname(os.path.abspath(__file__))
# ✅ models 폴더 접근 (RestormerVolterra import용)
sys.path.append(os.path.abspath(os.path.join(current_dir, '..', '..')))

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim

from models.restormer_volterra import RestormerVolterra


# ---------------- Dataset ----------------
class PairedRainDataset(Dataset):
    def __init__(self, input_dir, target_dir, transform=None):
        self.input_paths = natsorted(glob(os.path.join(input_dir, "*")))
        self.target_paths = natsorted(glob(os.path.join(target_dir, "*")))
        assert len(self.input_paths) == len(self.target_paths), \
            f"개수 불일치: {len(self.input_paths)} vs {len(self.target_paths)}"
        self.transform = transform

    def __len__(self): 
        return len(self.input_paths)

    def __getitem__(self, idx):
        inp = Image.open(self.input_paths[idx]).convert("RGB")
        tgt = Image.open(self.target_paths[idx]).convert("RGB")
        if self.transform:
            inp = self.transform(inp)
            tgt = self.transform(tgt)
        return inp, tgt


# ---------------- Evaluate ----------------
@torch.no_grad()
def evaluate(model, dataloader, device, tag=""):
    model.eval()
    total_psnr = total_ssim = 0.0
    n = 0

    for inp, tgt in dataloader:
        inp, tgt = inp.to(device), tgt.to(device)
        out = model(inp).clamp(0, 1)

        out_np = out[0].cpu().numpy().transpose(1, 2, 0)
        tgt_np = tgt[0].cpu().numpy().transpose(1, 2, 0)

        psnr = compute_psnr(tgt_np, out_np, data_range=1.0)
        ssim = compute_ssim(tgt_np, out_np, channel_axis=2, data_range=1.0)

        total_psnr += psnr
        total_ssim += ssim
        n += 1

    avg_psnr = total_psnr / n
    avg_ssim = total_ssim / n
    print(f"\n📊 Test on {tag}")
    print(f"Images: {n} | PSNR: {avg_psnr:.2f} dB | SSIM: {avg_ssim:.4f}")


# ---------------- Main ----------------
def main():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    CKPT_PATH = r"E:\restormer+volterra\checkpoints\restormer_volterra_csd\epoch_5_ssim0.9531_psnr33.03.pth"

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    test_sets = {
        "Rain100H": PairedRainDataset(
            r"E:/restormer+volterra/data/rain100H/test/rain",
            r"E:/restormer+volterra/data/rain100H/test/norain",
            transform
        ),
        "Rain100L": PairedRainDataset(
            r"E:/restormer+volterra/data/rain100L/test/rain",
            r"E:/restormer+volterra/data/rain100L/test/norain",
            transform
        ),
    }

    # 모델 로드
    model = RestormerVolterra().to(DEVICE)
    state = torch.load(CKPT_PATH, map_location=DEVICE)
    model.load_state_dict(state, strict=False)
    print(f"✓ Loaded checkpoint: {CKPT_PATH}")

    # 평가
    for tag, dataset in test_sets.items():
        dl = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
        evaluate(model, dl, DEVICE, tag)


if __name__ == "__main__":
    main()
 """

# Denoising
""" # E:/restormer+volterra/Restormer + Volterra/tasks/denoising/test_sidd.py
import os, sys
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
    CKPT_PATH = r"E:\restormer+volterra\checkpoints\restormer_volterra_csd\epoch_5_ssim0.9531_psnr33.03.pth"

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
    main()
 """

# Deblurring
""" 
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
    CKPT_PATH      = r"E:\restormer+volterra\checkpoints\restormer_volterra_csd\epoch_5_ssim0.9531_psnr33.03.pth"

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
    main() """


# KADID seperate
""" import os, sys
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
class PairedListDataset(Dataset):
    def __init__(self, list_path, transform):
        self.transform = transform
        self.pairs = []
        base_dir = os.path.dirname(list_path)
        with open(list_path, "r", encoding="utf-8") as f:
            for ln in f:
                ln = ln.strip()
                if not ln or ln.startswith("#"):
                    continue
                parts = [p.strip().strip('"').strip("'") for p in (ln.split(",") if "," in ln else ln.split())]
                if len(parts) >= 2:
                    dist_path = os.path.join(base_dir, parts[0]).replace("\\", "/")
                    ref_path  = os.path.join(base_dir, parts[1]).replace("\\", "/")
                    if os.path.exists(dist_path) and os.path.exists(ref_path):
                        self.pairs.append((dist_path, ref_path))

    def __len__(self): return len(self.pairs)

    def __getitem__(self, idx):
        inp, tgt = self.pairs[idx]
        inp_img = Image.open(inp).convert("RGB")
        tgt_img = Image.open(tgt).convert("RGB")
        name = os.path.basename(inp)
        return self.transform(inp_img), self.transform(tgt_img), name


# ---------------- Utils ----------------
def _to_numpy01(t: torch.Tensor) -> np.ndarray:
    if t.dim() == 4: 
        t = t[0]
    arr = t.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    return np.clip(arr, 0.0, 1.0)

def _save_img01(path: str, arr01: np.ndarray):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img = (arr01 * 255.0 + 0.5).astype(np.uint8)
    Image.fromarray(img).save(path)

@torch.no_grad()
def evaluate_and_save(model, dataloader, device, tag: str, save_root: str):
    model.eval()
    total_psnr = 0.0
    total_ssim = 0.0
    n = 0
    for x, y, names in tqdm(dataloader, desc=f"[Eval] {tag}"):
        x, y = x.to(device), y.to(device)
        out = model(x).clamp(0, 1)

        # 저장
        for i in range(out.size(0)):
            out_np = _to_numpy01(out[i])
            stem = os.path.splitext(names[i])[0]
            save_path = os.path.join(save_root, tag, f"{stem}.png")
            _save_img01(save_path, out_np)

        # 메트릭
        out_np = out.cpu().numpy()
        gt_np  = y.cpu().numpy()
        b = out_np.shape[0]
        for i in range(b):
            o = np.transpose(out_np[i], (1, 2, 0))
            g = np.transpose(gt_np[i],  (1, 2, 0))
            total_psnr += compute_psnr(g, o, data_range=1.0)
            total_ssim += compute_ssim(g, o, channel_axis=2, data_range=1.0)
            n += 1

    return (total_psnr/n if n else 0), (total_ssim/n if n else 0), n


# ---------------- Main ----------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ✅ 경로
    BASE_DIR = r"E:/restormer+volterra/data/kadid_seperate"
    CKPT = r"E:\restormer+volterra\checkpoints\restormer_volterra_csd\epoch_5_ssim0.9531_psnr33.03.pth"
    RESULT_DIR = r"E:/restormer+volterra/results/kadid"
    os.makedirs(RESULT_DIR, exist_ok=True)

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    # ✅ Dataset 준비
    test_sets = {
        "KADID-gaussian": PairedListDataset(
            os.path.join(BASE_DIR, "gaussian", "pairs_gaussian.txt"), transform),
        "KADID-impulse": PairedListDataset(
            os.path.join(BASE_DIR, "impulse noise", "pairs_impulse.txt"), transform),
        "KADID-white": PairedListDataset(
            os.path.join(BASE_DIR, "white noise", "pairs_white.txt"), transform),
    }

    # ✅ 모델 로드
    model = RestormerVolterra().to(device)
    state = torch.load(CKPT, map_location=device)
    model.load_state_dict(state, strict=False)
    print(f"✓ Loaded checkpoint: {CKPT}")

    # ✅ 평가
    print("\n=========== KADID Test Summary ===========")
    for tag, dataset in test_sets.items():
        dl = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)
        psnr, ssim, n = evaluate_and_save(model, dl, device, tag, RESULT_DIR)
        print(f"{tag:15s} | Images: {n:4d} | PSNR: {psnr:.2f} dB | SSIM: {ssim:.4f}")


if __name__ == "__main__":
    main()
 """

# SOTS
import os, sys, csv
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
class PairedCSVDataset(Dataset):
    """
    CSV 파일에 (dist_img, ref_img) 절대경로 저장된 경우 사용
    """
    def __init__(self, csv_path, transform):
        import pandas as pd
        self.df = pd.read_csv(csv_path)
        self.transform = transform

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        inp = Image.open(self.df.iloc[idx, 0]).convert("RGB")
        tgt = Image.open(self.df.iloc[idx, 1]).convert("RGB")
        name = os.path.basename(self.df.iloc[idx, 0])
        return self.transform(inp), self.transform(tgt), name


# ---------------- Utils ----------------
def _to_numpy01(t: torch.Tensor) -> np.ndarray:
    if t.dim() == 4:
        t = t[0]
    arr = t.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    return np.clip(arr, 0.0, 1.0)

def _save_img01(path: str, arr01: np.ndarray):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img = (arr01 * 255.0 + 0.5).astype(np.uint8)
    Image.fromarray(img).save(path)

@torch.no_grad()
def evaluate_and_save(model, dataloader, device, tag: str, save_root: str):
    model.eval()
    total_psnr = 0.0
    total_ssim = 0.0
    n = 0
    for x, y, names in tqdm(dataloader, desc=f"[Eval] {tag}"):
        x, y = x.to(device), y.to(device)
        out = model(x).clamp(0, 1)

        # 저장
        for i in range(out.size(0)):
            out_np = _to_numpy01(out[i])
            stem = os.path.splitext(names[i])[0]
            save_path = os.path.join(save_root, tag, f"{stem}.png")
            _save_img01(save_path, out_np)

        # 메트릭
        out_np = out.cpu().numpy()
        gt_np  = y.cpu().numpy()
        b = out_np.shape[0]
        for i in range(b):
            o = np.transpose(out_np[i], (1, 2, 0))
            g = np.transpose(gt_np[i],  (1, 2, 0))
            total_psnr += compute_psnr(g, o, data_range=1.0)
            total_ssim += compute_ssim(g, o, channel_axis=2, data_range=1.0)
            n += 1

    return (total_psnr/n if n else 0), (total_ssim/n if n else 0), n


# ---------------- Main ----------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ✅ 경로
    BASE_DIR = r"E:/restormer+volterra/data/SOTS"
    SPLIT_DIR = os.path.join(BASE_DIR, "splits")
    CKPT = r"E:\restormer+volterra\checkpoints\restormer_volterra_csd\epoch_5_ssim0.9531_psnr33.03.pth"
    RESULT_DIR = r"E:/restormer+volterra/results/sots"
    os.makedirs(RESULT_DIR, exist_ok=True)

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    # ✅ Dataset 준비
    test_sets = {
        "SOTS-indoor": PairedCSVDataset(os.path.join(SPLIT_DIR, "indoor_test.csv"), transform),
        "SOTS-outdoor": PairedCSVDataset(os.path.join(SPLIT_DIR, "outdoor_test.csv"), transform),
    }

    # ✅ 모델 로드
    model = RestormerVolterra().to(device)
    state = torch.load(CKPT, map_location=device)
    model.load_state_dict(state, strict=False)
    print(f"✓ Loaded checkpoint: {CKPT}")

    # ✅ 평가
    print("\n=========== SOTS Test Summary ===========")
    for tag, dataset in test_sets.items():
        dl = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)
        psnr, ssim, n = evaluate_and_save(model, dl, device, tag, RESULT_DIR)
        print(f"{tag:12s} | Images: {n:4d} | PSNR: {psnr:.2f} dB | SSIM: {ssim:.4f}")
    print(f"\n🖼 Restored images saved under: {RESULT_DIR}")


if __name__ == "__main__":
    main()
