# VTN unified benchmark test script
# Unified benchmark test script (PSNR/SSIM 異쒕젰 + (?듭뀡) triplet PNG ???
# - Rain100H / Rain100L / GoPro / RESIDE-6K / CSD
# - AMP 吏??
# - (?듭뀡) tile inference 吏??OOM 諛⑹?)
#
# Project layout:
#   - models/restormer_volterra.py
#   - datasets/*.py
#   - Run from the VTN project root or scripts folder.

import os
import sys
import math
import argparse
from typing import Tuple, Optional

import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from torch.amp import autocast
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim

from PIL import Image, ImageDraw, ImageFont


# ----------------------
# Path setup (models/datasets/config: VTN root)
# ----------------------
CUR_DIR = os.path.dirname(os.path.abspath(__file__))  # .../VTN/scripts
ROOT_DIR = os.path.dirname(CUR_DIR)                   # .../VTN
for p in [CUR_DIR, ROOT_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from models.restormer_volterra import RestormerVolterra
from datasets.rain100h_dataset import Rain100HDataset
from datasets.rain100l_dataset import Rain100LDataset
from datasets.gopro_dataset import GoProDataset
from datasets.reside_dataset import RESIDEDataset
from datasets.csd_dataset import CSDDataset
from config import DATA, CHECKPOINTS, result_dir, as_str


# ======================
# Config
# ======================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = True

# ????理쒖쥌 pth濡?援먯껜?댁꽌 ?ъ슜
CKPT_KEY = "mixed7"
CKPT_PATH = as_str(CHECKPOINTS[CKPT_KEY])

# ???됯? ?낅젰 ?댁긽??(?듭씪)
EVAL_SIZE = 256

# ??Tile inference (OOM 諛⑹?). 蹂댄넻 0?대㈃ 鍮꾪솢?? 256~512 異붿쿇
USE_TILE = False
TILE_SIZE = 256
TILE_OVERLAP = 32

# ??寃곌낵 ???
SAVE_TRIPLETS = True
SAVE_N_PER_DATASET = 5
RESULTS_DIR = as_str(result_dir("unified_test"))
os.makedirs(RESULTS_DIR, exist_ok=True)

# ??dataset paths
RAIN100H_TEST = as_str(DATA["rain100h"] / "test")
RAIN100L_TEST = as_str(DATA["rain100l"] / "test")
GOPRO_CSV_TEST = as_str(DATA["gopro"] / "gopro_test_pairs.csv")
RESIDE_ROOT = as_str(DATA["reside"])
CSD_ROOT = as_str(DATA["csd"])


# ======================
# Utils
# ======================
def tensor_to_hwc01(x: torch.Tensor) -> np.ndarray:
    """(1,C,H,W) or (C,H,W) -> HWC float32 [0,1]"""
    if x.dim() == 4:
        x = x[0]
    x = x.detach().float().clamp(0, 1).cpu().numpy()
    return np.transpose(x, (1, 2, 0))


def hwc01_to_pil(img: np.ndarray) -> Image.Image:
    """HWC [0,1] -> PIL RGB"""
    img_u8 = (np.clip(img, 0, 1) * 255.0).astype(np.uint8)
    return Image.fromarray(img_u8, mode="RGB")


def render_triplet_with_text(inp_np, restored_np, gt_np, psnr, ssim,
                             title_left="Input", title_mid="Restored", title_right="GT",
                             pad=12, text_h=54) -> Image.Image:
    inp_p = hwc01_to_pil(inp_np)
    res_p = hwc01_to_pil(restored_np)
    gt_p = hwc01_to_pil(gt_np)

    w, h = inp_p.size
    canvas_w = w * 3 + pad * 4
    canvas_h = h + text_h + pad * 3

    canvas = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    try:
        font_b = ImageFont.truetype("arial.ttf", 20)
    except Exception:
        font_b = ImageFont.load_default()

    y_title = pad
    draw.text((pad + (w // 2) - 30, y_title), title_left, fill=(0, 0, 0), font=font_b)
    draw.text((pad * 2 + w + (w // 2) - 42, y_title), title_mid, fill=(0, 0, 0), font=font_b)
    draw.text((pad * 3 + w * 2 + (w // 2) - 10, y_title), title_right, fill=(0, 0, 0), font=font_b)

    y_img = pad * 2 + 18
    x1 = pad
    x2 = pad * 2 + w
    x3 = pad * 3 + w * 2

    canvas.paste(inp_p, (x1, y_img))
    canvas.paste(res_p, (x2, y_img))
    canvas.paste(gt_p, (x3, y_img))

    y_text = y_img + h + pad
    metric_text = f"PSNR: {psnr:.2f} dB    SSIM: {ssim:.4f}"
    draw.text((pad, y_text), metric_text, fill=(0, 0, 0), font=font_b)

    return canvas


@torch.no_grad()
def forward_with_optional_tile(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    """
    x: (1,C,H,W) in [0,1]
    return: (1,C,H,W)
    """
    if (not USE_TILE) or (x.shape[-1] <= TILE_SIZE and x.shape[-2] <= TILE_SIZE):
        with autocast(device_type="cuda", enabled=(USE_AMP and DEVICE.type == "cuda")):
            return model(x)

    b, c, h, w = x.shape
    tile = TILE_SIZE
    overlap = TILE_OVERLAP
    stride = tile - overlap
    if stride <= 0:
        raise ValueError("TILE_OVERLAP must be smaller than TILE_SIZE")

    out = torch.zeros_like(x)
    weight = torch.zeros((1, 1, h, w), device=x.device, dtype=x.dtype)

    # simple cosine-ish weight (center emphasis)
    yy = torch.linspace(0, math.pi, tile, device=x.device, dtype=x.dtype)
    win1d = (0.5 - 0.5 * torch.cos(yy)).view(1, 1, tile, 1)
    xx = torch.linspace(0, math.pi, tile, device=x.device, dtype=x.dtype)
    win2d = win1d * (0.5 - 0.5 * torch.cos(xx)).view(1, 1, 1, tile)

    for top in range(0, h, stride):
        for left in range(0, w, stride):
            bottom = min(top + tile, h)
            right = min(left + tile, w)
            top2 = max(bottom - tile, 0)
            left2 = max(right - tile, 0)

            patch = x[:, :, top2:bottom, left2:right]
            with autocast(device_type="cuda", enabled=(USE_AMP and DEVICE.type == "cuda")):
                pred = model(patch)

            wh = bottom - top2
            ww = right - left2
            win = win2d[:, :, :wh, :ww]

            out[:, :, top2:bottom, left2:right] += pred * win
            weight[:, :, top2:bottom, left2:right] += win

    out = out / (weight + 1e-8)
    return out


def build_loader_for_dataset(name: str):
    tfm = transforms.Compose([
        transforms.Resize((EVAL_SIZE, EVAL_SIZE)),
        transforms.ToTensor(),
    ])

    if name == "Rain100H":
        ds = Rain100HDataset(root_dir=RAIN100H_TEST, transform=tfm)
    elif name == "Rain100L":
        ds = Rain100LDataset(root_dir=RAIN100L_TEST, transform=tfm)
    elif name == "GoPro":
        # ????GoProDataset ?쒓렇?덉쿂: GoProDataset(csv_file, transform=None)
        ds = GoProDataset(GOPRO_CSV_TEST, transform=tfm)
    elif name == "RESIDE":
        ds = RESIDEDataset(root_dir=RESIDE_ROOT, split="test", transform=tfm, strict=True)
    elif name == "CSD":
        ds = CSDDataset(root_dir=CSD_ROOT, split="test", transform=tfm)
    else:
        raise ValueError(f"Unknown dataset name: {name}")

    loader = DataLoader(
        ds,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=(DEVICE.type == "cuda"),
    )
    return ds, loader


@torch.no_grad()
def eval_one_dataset(model, name: str) -> Tuple[float, float]:
    ds, loader = build_loader_for_dataset(name)

    save_dir = os.path.join(RESULTS_DIR, name)
    if SAVE_TRIPLETS:
        os.makedirs(save_dir, exist_ok=True)

    psnr_sum, ssim_sum, n = 0.0, 0.0, 0
    saved = 0

    pbar = tqdm(loader, desc=f"Testing {name}", leave=False)
    for i, (inp, gt) in enumerate(pbar, start=1):
        inp = inp.to(DEVICE, non_blocking=True)
        gt = gt.to(DEVICE, non_blocking=True)

        out = forward_with_optional_tile(model, inp)

        out_np = tensor_to_hwc01(out)
        inp_np = tensor_to_hwc01(inp)
        gt_np = tensor_to_hwc01(gt)

        psnr = compute_psnr(gt_np, out_np, data_range=1.0)
        ssim = compute_ssim(gt_np, out_np, channel_axis=2, data_range=1.0, win_size=7)

        psnr_sum += psnr
        ssim_sum += ssim
        n += 1

        pbar.set_postfix(psnr=f"{psnr:.2f}", ssim=f"{ssim:.4f}")

        if SAVE_TRIPLETS and saved < SAVE_N_PER_DATASET:
            vis = render_triplet_with_text(
                inp_np, out_np, gt_np, psnr, ssim,
                title_left="Input", title_mid="Restored", title_right="GT"
            )
            out_path = os.path.join(save_dir, f"sample_{saved+1:02d}_psnr{psnr:.2f}_ssim{ssim:.4f}.png")
            vis.save(out_path)
            saved += 1

    avg_psnr = psnr_sum / max(n, 1)
    avg_ssim = ssim_sum / max(n, 1)

    print(f"\n[{name}] N={n}")
    print(f"  ??PSNR: {avg_psnr:.2f} dB")
    print(f"  ??SSIM: {avg_ssim:.4f}")
    if SAVE_TRIPLETS:
        print(f"  ?뼹截? Triplets saved: {min(saved, SAVE_N_PER_DATASET)} -> {save_dir}")

    return avg_psnr, avg_ssim


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate VTN all-in-one checkpoints.")
    parser.add_argument(
        "--ckpt-key",
        default=CKPT_KEY,
        choices=sorted(CHECKPOINTS.keys()),
        help="Checkpoint key defined in config.py.",
    )
    parser.add_argument(
        "--ckpt",
        default=None,
        help="Optional explicit checkpoint path. Overrides --ckpt-key.",
    )
    parser.add_argument(
        "--result-name",
        default=None,
        help="Optional result subfolder name under VTN/results.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    ckpt_path = args.ckpt if args.ckpt else as_str(CHECKPOINTS[args.ckpt_key])
    results_dir = as_str(result_dir(args.result_name or f"unified_test_{args.ckpt_key}"))

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    print(f"[Device] {DEVICE}")
    print(f"[CKPT ] {ckpt_path}")
    print(f"[Eval ] size={EVAL_SIZE} | AMP={USE_AMP} | TILE={USE_TILE} (tile={TILE_SIZE}, overlap={TILE_OVERLAP})")
    print(f"[Save ] RESULTS_DIR={results_dir} | SAVE_TRIPLETS={SAVE_TRIPLETS} | N={SAVE_N_PER_DATASET}")

    model = RestormerVolterra().to(DEVICE)
    state = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(state, strict=True)
    model.eval()

    # ---- evaluate all ----
    global RESULTS_DIR
    RESULTS_DIR = results_dir

    names = ["Rain100H", "Rain100L", "GoPro", "RESIDE", "CSD"]
    results = {}
    for name in names:
        psnr, ssim = eval_one_dataset(model, name)
        results[name] = (psnr, ssim)

    # ---- summary ----
    rain_psnr = (results["Rain100H"][0] + results["Rain100L"][0]) / 2.0
    rain_ssim = (results["Rain100H"][1] + results["Rain100L"][1]) / 2.0

    avg_psnr = (rain_psnr + results["GoPro"][0] + results["RESIDE"][0] + results["CSD"][0]) / 4.0
    avg_ssim = (rain_ssim + results["GoPro"][1] + results["RESIDE"][1] + results["CSD"][1]) / 4.0

    print("\n==============================")
    print("?뱤 Unified Test Summary")
    print(f"Rain100H : PSNR {results['Rain100H'][0]:.2f} | SSIM {results['Rain100H'][1]:.4f}")
    print(f"Rain100L : PSNR {results['Rain100L'][0]:.2f} | SSIM {results['Rain100L'][1]:.4f}")
    print(f"Rain(avg): PSNR {rain_psnr:.2f} | SSIM {rain_ssim:.4f}")
    print(f"GoPro    : PSNR {results['GoPro'][0]:.2f} | SSIM {results['GoPro'][1]:.4f}")
    print(f"RESIDE   : PSNR {results['RESIDE'][0]:.2f} | SSIM {results['RESIDE'][1]:.4f}")
    print(f"CSD      : PSNR {results['CSD'][0]:.2f} | SSIM {results['CSD'][1]:.4f}")
    print("------------------------------")
    print(f"AVG(4 tasks: Rain+GoPro+RESIDE+CSD)")
    print(f"??PSNR {avg_psnr:.2f} dB | ??SSIM {avg_ssim:.4f}")
    print("==============================\n")


if __name__ == "__main__":
    main()

