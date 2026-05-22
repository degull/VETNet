import os, sys, random
from typing import List, Tuple, Dict

current_dir = os.path.dirname(os.path.abspath(__file__))
# models 폴더 접근 (RestormerVolterra import용)
sys.path.append(os.path.abspath(os.path.join(current_dir, '..', '..')))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from torch.amp import autocast, GradScaler

from models.restormer_volterra import RestormerVolterra

# ---------------- Paths ----------------
BASE_DIR   = r"E:/restormer+volterra/data/kadid_seperate/white noise"
PAIRS_TXT  = os.path.join(BASE_DIR, "pairs_white.txt")
SAVE_DIR   = r"E:/restormer+volterra/checkpoints/kadid_white_noise"
os.makedirs(SAVE_DIR, exist_ok=True)

# ---------------- Train config ----------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH  = 4
EPOCHS = 100
LR     = 2e-4
SEED   = 42
TEST_RATIO = 0.2

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if DEVICE.type == "cuda":
    torch.cuda.manual_seed_all(SEED)

# Resize 스케줄 (선형 curriculum)
resize_schedule = {0: 128, 30: 192, 60: 256}
def get_transform(epoch: int):
    size = max(v for k, v in resize_schedule.items() if epoch >= k)
    return transforms.Compose([
        transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor()
    ])

# ---------------- I/O utils ----------------
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
    """
    GT stem(I01 등) 단위로 train/test 분리 → 누수 방지
    """
    # stem -> list of pairs
    bucket: Dict[str, List[Tuple[str, str]]] = {}
    for dist, gt in pairs:
        stem = os.path.splitext(os.path.basename(gt))[0]  # 'I01'
        bucket.setdefault(stem, []).append((dist, gt))

    stems = sorted(bucket.keys())
    random.Random(seed).shuffle(stems)

    n_test = max(1, int(len(stems) * test_ratio))
    test_stems = set(stems[:n_test])

    train_pairs, test_pairs = [], []
    for stem, group in bucket.items():
        if stem in test_stems:
            test_pairs.extend(group)
        else:
            train_pairs.extend(group)

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

# ---------------- Eval ----------------
@torch.no_grad()
def evaluate(model, dl):
    model.eval()
    P = S = 0.0
    n = 0
    for x, y in tqdm(dl, desc="[Eval]", leave=False):
        x, y = x.to(DEVICE), y.to(DEVICE)
        with autocast(device_type="cuda", enabled=(DEVICE.type=="cuda")):
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
    return (P/n if n else 0.0), (S/n if n else 0.0)

# ---------------- Main ----------------
def main():
    print("🚀 KADID white noise training (no leakage by GT stem)")
    # 1) 페어 읽기 & split
    all_pairs = read_pairs_txt(BASE_DIR, PAIRS_TXT)
    train_pairs, test_pairs = split_by_gt_stem(all_pairs, TEST_RATIO, SEED)
    print(f"[Pairs] total={len(all_pairs)}, train={len(train_pairs)}, test={len(test_pairs)}")

    # 2) 모델/옵티마이저
    model  = RestormerVolterra().to(DEVICE)
    opt    = optim.AdamW(model.parameters(), lr=LR)
    crit   = nn.L1Loss()
    scaler = GradScaler()

    for e in range(1, EPOCHS+1):
        tfm_tr = get_transform(e)
        tr_ds  = PairFileDataset(train_pairs, tfm_tr)
        tr_dl  = DataLoader(tr_ds, batch_size=BATCH, shuffle=True, num_workers=2, pin_memory=True)

        # ----- Train -----
        model.train()
        tot = 0.0
        loop = tqdm(tr_dl, desc=f"[Train] {e}/{EPOCHS}")
        opt.zero_grad(set_to_none=True)
        for x, y in loop:
            x, y = x.to(DEVICE), y.to(DEVICE)
            with autocast(device_type="cuda", enabled=(DEVICE.type=="cuda")):
                out  = model(x)
                loss = crit(out, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            tot += loss.item()
            loop.set_postfix(loss=f"{loss.item():.4f}")
        avg_train = tot / max(1, len(tr_dl))

        # ----- Validation @ 256 -----
        tfm_val = transforms.Compose([
            transforms.Resize((256,256), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor()
        ])
        va_ds = PairFileDataset(test_pairs, tfm_val)
        va_dl = DataLoader(va_ds, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)
        vp, vs = evaluate(model, va_dl)

        print(f"📣 Epoch {e:03d} | TrainLoss {avg_train:.5f} | Val PSNR {vp:.2f} | Val SSIM {vs:.4f}")

        ckpt = os.path.join(SAVE_DIR, f"epoch_{e}_valssim{vs:.4f}_valpsnr{vp:.2f}.pth")
        torch.save(model.state_dict(), ckpt)
        print(f"💾 Saved: {ckpt}")

if __name__ == "__main__":
    main()
