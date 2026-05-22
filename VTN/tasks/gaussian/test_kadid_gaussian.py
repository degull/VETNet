import os, sys
from typing import List, Tuple, Dict

current_dir = os.path.dirname(os.path.abspath(__file__))
# models 폴더 접근 (RestormerVolterra import용)
sys.path.append(os.path.abspath(os.path.join(current_dir, '..', '..')))

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from models.restormer_volterra import RestormerVolterra

# ---------------- Paths ----------------
BASE_DIR   = r"E:/restormer+volterra/data/kadid_seperate/gaussian"
PAIRS_TXT  = os.path.join(BASE_DIR, "pairs_gaussian.txt")
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ 평가할 체크포인트 경로로 바꿔주세요
CKPT = r"E:\restormer+volterra\checkpoints\sots_volterra\epoch_98_valssim0.9573_valpsnr26.62.pth"

# ---------------- Split Utils (train과 동일 로직) ----------------
def read_pairs_txt(base_dir: str, txt_path: str) -> List[Tuple[str, str]]:
    pairs = []
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            dist_rel, gt_rel = line.split()
            pairs.append((
                os.path.join(base_dir, dist_rel.replace("/", os.sep)),
                os.path.join(base_dir, gt_rel.replace("/", os.sep))
            ))
    return pairs

def split_by_gt_stem(pairs: List[Tuple[str, str]], test_ratio: float, seed: int):
    bucket: Dict[str, List[Tuple[str, str]]] = {}
    for dist, gt in pairs:
        stem = os.path.splitext(os.path.basename(gt))[0]
        bucket.setdefault(stem, []).append((dist, gt))
    stems = sorted(bucket.keys())
    import random
    random.Random(seed).shuffle(stems)
    n_test = max(1, int(len(stems) * test_ratio))
    test_stems = set(stems[:n_test])
    train_pairs, test_pairs = [], []
    for stem, group in bucket.items():
        (test_pairs if stem in test_stems else train_pairs).extend(group)
    return train_pairs, test_pairs

# ---------------- Dataset ----------------
class PairFileDataset(Dataset):
    def __init__(self, pairs: List[Tuple[str, str]], tfm):
        self.pairs = pairs
        self.tfm = tfm
    def __len__(self):
        return len(self.pairs)
    def __getitem__(self, idx):
        dist_path, gt_path = self.pairs[idx]
        x = Image.open(dist_path).convert("RGB")
        y = Image.open(gt_path).convert("RGB")
        return self.tfm(x), self.tfm(y)

@torch.no_grad()
def evaluate(model, dl):
    model.eval()
    P = S = 0.0
    n = 0
    for x, y in tqdm(dl, desc="[Test Eval]"):
        x, y = x.to(DEVICE), y.to(DEVICE)
        out = model(x).clamp(0,1)
        out_np = out.cpu().numpy()
        gt_np  = y.cpu().numpy()
        b = out_np.shape[0]
        for i in range(b):
            O = np.transpose(out_np[i], (1,2,0))
            G = np.transpose(gt_np[i],  (1,2,0))
            P += psnr(G, O, data_range=1.0)
            S += ssim(G, O, channel_axis=2, data_range=1.0)
            n += 1
    return (P/n if n else 0.0), (S/n if n else 0.0), n

def main():
    # Train과 동일한 방식/시드로 split → test만 사용 (누수 방지)
    TEST_RATIO = 0.2
    SEED = 42

    all_pairs = read_pairs_txt(BASE_DIR, PAIRS_TXT)
    _, test_pairs = split_by_gt_stem(all_pairs, TEST_RATIO, SEED)
    print(f"[Test Pairs] total={len(all_pairs)} | test={len(test_pairs)}")

    tfm = transforms.Compose([
        transforms.Resize((256,256), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor()
    ])
    te_ds = PairFileDataset(test_pairs, tfm)
    te_dl = DataLoader(te_ds, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)

    # 모델 로드
    model = RestormerVolterra().to(DEVICE)
    state = torch.load(CKPT, map_location=DEVICE)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    state = {k.replace("module.",""): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    print(f"✓ Loaded checkpoint: {CKPT}")

    # 평가
    p, s, n = evaluate(model, te_dl)
    print("\n=========== KADID Gaussian Test ===========")
    print(f"Images: {n:4d} | PSNR: {p:.2f} dB | SSIM: {s:.4f}")
    print("===========================================")

if __name__ == "__main__":
    main()
