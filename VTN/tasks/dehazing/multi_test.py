# E:/restormer+volterra/Restormer + Volterra/tasks/dehazing/multi_test.py
import os, sys, glob
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim

# ----- model import path -----
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(current_dir, '..', '..')))
from models.restormer_volterra import RestormerVolterra

# ----- config -----
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CKPT = r"E:/restormer+volterra/checkpoints/sots_volterra/epoch_77_valssim0.9580_valpsnr26.83.pth"
RESULT_ROOT = r"E:/restormer+volterra/results/unified_eval"
EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif")

os.makedirs(RESULT_ROOT, exist_ok=True)

# ----- datasets -----
class PairedFolderDataset(Dataset):
    """(input_dir, target_dir)에서 파일명 매칭"""
    def __init__(self, input_dir, target_dir, return_paths=False):
        self.input_paths = sorted(
            [p for p in glob.glob(os.path.join(input_dir, "*")) if os.path.splitext(p)[1].lower() in EXTS]
        )
        self.target_paths = [os.path.join(target_dir, os.path.basename(p)) for p in self.input_paths]
        self.return_paths = return_paths

    def __len__(self): return len(self.input_paths)

    def _load01(self, path):
        img = Image.open(path).convert("RGB")
        arr = np.float32(img) / 255.0
        return torch.from_numpy(arr).permute(2, 0, 1)  # CHW, [0,1]

    def __getitem__(self, idx):
        x = self._load01(self.input_paths[idx])
        y_path = self.target_paths[idx]
        y = self._load01(y_path) if os.path.exists(y_path) else torch.zeros_like(x)
        if self.return_paths:
            return x, y, self.input_paths[idx]
        return x, y

# ----- utils -----
def _to_numpy01(t: torch.Tensor) -> np.ndarray:
    # t: [C,H,W]
    return np.clip(t.detach().cpu().permute(1, 2, 0).numpy(), 0, 1)

def _save_img01(path: str, arr01: np.ndarray):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray((arr01 * 255).astype(np.uint8)).save(path)

@torch.no_grad()
def evaluate_and_save(model, dataloader, tag: str, save_root: str):
    """배치 차원은 이미 DataLoader가 붙여줌: x=[B,C,H,W]"""
    model.eval()
    psnr_tot, ssim_tot, n = 0.0, 0.0, 0
    factor = 8

    for batch in tqdm(dataloader, desc=f"[Eval] {tag}"):
        # batch unpack (paths는 list[str])
        if len(batch) == 3:
            x, y, paths = batch
        else:
            x, y = batch
            paths = [f"{tag}_{n+i:05d}.png" for i in range(x.size(0))]

        x, y = x.to(DEVICE), y.to(DEVICE)                  # [B,C,H,W]
        _, _, h, w = x.shape
        H, W = ((h + factor) // factor) * factor, ((w + factor) // factor) * factor
        x_pad = F.pad(x, (0, W - w, 0, H - h), mode="reflect")

        out = model(x_pad).clamp(0, 1)[:, :, :h, :w]       # [B,C,H,W]

        # save each image
        for i in range(out.size(0)):
            stem = os.path.splitext(os.path.basename(paths[i]))[0]
            save_path = os.path.join(save_root, tag, f"{stem}.png")
            _save_img01(save_path, _to_numpy01(out[i]))

        # metrics
        out_np, gt_np = out.cpu().numpy(), y.cpu().numpy()
        for i in range(out_np.shape[0]):
            O = np.transpose(out_np[i], (1, 2, 0))
            G = np.transpose(gt_np[i], (1, 2, 0))
            if G.sum() == 0:   # GT가 없으면 스킵
                continue
            psnr_tot += compute_psnr(G, O, data_range=1.0)
            ssim_tot += compute_ssim(G, O, channel_axis=2, data_range=1.0)
            n += 1

    psnr_avg = psnr_tot / n if n else 0.0
    ssim_avg = ssim_tot / n if n else 0.0
    return psnr_avg, ssim_avg, n

def main():
    # 평가할 데이터셋 경로들
    datasets = {
        "Rain100H": (r"E:/restormer+volterra/data/rain100H/test/rain",
                     r"E:/restormer+volterra/data/rain100H/test/norain"),
        "Rain100L": (r"E:/restormer+volterra/data/rain100L/test/rain",
                     r"E:/restormer+volterra/data/rain100L/test/norain"),
        "HIDE":     (r"E:/restormer+volterra/data/HIDE/test",
                     r"E:/restormer+volterra/data/HIDE/GT"),
        "SIDD":     (r"E:/restormer+volterra/data/SIDD/test/noisy",
                     r"E:/restormer+volterra/data/SIDD/test/gt"),
        "CSD":      (r"E:/restormer+volterra/data/CSD/Test/Snow",
                     r"E:/restormer+volterra/data/CSD/Test/Gt"),
        # KADID 분리 폴더 구조(사용자 환경에 맞춰 조정)
        "KADID_Gaussian": (r"E:/restormer+volterra/data/kadid_seperate/gaussian/blur/level_1",  # 예시 입력
                           r"E:/restormer+volterra/data/kadid_seperate/gaussian/gt"),           # GT
        "KADID_Impulse":  (r"E:/restormer+volterra/data/kadid_seperate/impulse/blur/level_1",
                           r"E:/restormer+volterra/data/kadid_seperate/impulse/gt"),
        "KADID_White":    (r"E:/restormer+volterra/data/kadid_seperate/white/blur/level_1",
                           r"E:/restormer+volterra/data/kadid_seperate/white/gt"),
    }

    # load model
    model = RestormerVolterra().to(DEVICE)
    state = torch.load(CKPT, map_location=DEVICE)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    print(f"✓ Loaded checkpoint: {CKPT}")

    # eval
    print("\n=========== Unified Evaluation ===========")
    for tag, (inp_dir, tgt_dir) in datasets.items():
        if not (os.path.isdir(inp_dir) and os.path.isdir(tgt_dir)):
            print(f"[Skip] {tag}: path not found → {inp_dir} | {tgt_dir}")
            continue
        ds = PairedFolderDataset(inp_dir, tgt_dir, return_paths=True)
        if len(ds) == 0:
            print(f"[Skip] {tag}: no images")
            continue
        dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)
        psnr, ssim, n = evaluate_and_save(model, dl, tag, RESULT_ROOT)
        print(f"{tag:12s} | images: {n:4d} | PSNR: {psnr:.2f} dB | SSIM: {ssim:.4f}")
    print("==========================================")

if __name__ == "__main__":
    main()
