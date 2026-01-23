import os
import sys
import glob
import math
import argparse
from typing import Tuple, Optional, List

import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# skimage for metrics
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim


# -------------------------
# Path / import safety
# -------------------------
CUR_DIR = os.path.dirname(os.path.abspath(__file__))
if CUR_DIR not in sys.path:
    sys.path.insert(0, CUR_DIR)

from models.restormer_volterra import RestormerVolterra

# -------------------------
# Utils: image I/O
# -------------------------
def _to_rgb_pil(img: Image.Image) -> Image.Image:
    if img.mode == "RGB":
        return img
    return img.convert("RGB")

def load_image_rgb(path: str) -> Image.Image:
    img = Image.open(path)
    img = _to_rgb_pil(img)
    return img

def pil_to_tensor_01(img: Image.Image) -> torch.Tensor:
    # RGB, [0,1], shape: (1,3,H,W)
    arr = np.array(img).astype(np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))  # C,H,W
    ten = torch.from_numpy(arr).unsqueeze(0)  # 1,C,H,W
    return ten

def tensor_01_to_pil(x: torch.Tensor) -> Image.Image:
    # x: (1,3,H,W) or (3,H,W), [0,1]
    if x.dim() == 4:
        x = x[0]
    x = x.detach().float().cpu().clamp(0, 1).numpy()
    x = np.transpose(x, (1, 2, 0))  # H,W,C
    x = (x * 255.0 + 0.5).astype(np.uint8)
    return Image.fromarray(x, mode="RGB")

def ensure_dir(d: str):
    os.makedirs(d, exist_ok=True)


# -------------------------
# Metrics
# -------------------------
def psnr_ssim(ref_rgb: np.ndarray, est_rgb: np.ndarray) -> Tuple[float, float]:
    """
    ref_rgb, est_rgb: float32, HxWx3, range [0,1]
    """
    psnr = compute_psnr(ref_rgb, est_rgb, data_range=1.0)
    ssim = compute_ssim(ref_rgb, est_rgb, channel_axis=2, data_range=1.0, win_size=7)
    return float(psnr), float(ssim)


# -------------------------
# Drawing
# -------------------------
def get_font(font_size: int = 22) -> ImageFont.ImageFont:
    # Try common fonts; fallback to default
    candidates = [
        "arial.ttf",
        "Arial.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/ARIAL.TTF",
        "C:/Windows/Fonts/calibri.ttf",
    ]
    for fp in candidates:
        try:
            return ImageFont.truetype(fp, font_size)
        except Exception:
            pass
    return ImageFont.load_default()

def draw_label_block(
    canvas: Image.Image,
    x0: int,
    y0: int,
    title: str,
    lines: List[str],
    font: ImageFont.ImageFont,
    pad: int = 10
):
    draw = ImageDraw.Draw(canvas)
    # measure
    all_text = [title] + lines
    widths, heights = [], []
    for t in all_text:
        bbox = draw.textbbox((0, 0), t, font=font)
        widths.append(bbox[2] - bbox[0])
        heights.append(bbox[3] - bbox[1])
    w = max(widths) + 2 * pad
    h = sum(heights) + (len(heights) + 1) * 4 + 2 * pad

    # background box (semi-opaque effect by drawing solid rectangle)
    rect = (x0, y0, x0 + w, y0 + h)
    draw.rectangle(rect, fill=(0, 0, 0))

    # text
    y = y0 + pad
    draw.text((x0 + pad, y), title, fill=(255, 255, 255), font=font)
    y += heights[0] + 8
    for i, ln in enumerate(lines):
        draw.text((x0 + pad, y), ln, fill=(255, 255, 255), font=font)
        y += heights[i + 1] + 6


def make_triplet_canvas(
    inp_img: Image.Image,
    out_img: Image.Image,
    gt_img: Image.Image,
    inp_metrics: Tuple[float, float],
    out_metrics: Tuple[float, float],
    save_path: str
):
    """
    Creates a single image: [input | restored | gt]
    Adds PSNR/SSIM for input-vs-gt and restored-vs-gt inside the image.
    """
    inp_img = _to_rgb_pil(inp_img)
    out_img = _to_rgb_pil(out_img)
    gt_img = _to_rgb_pil(gt_img)

    # unify sizes (use input size as target)
    W, H = inp_img.size
    if out_img.size != (W, H):
        out_img = out_img.resize((W, H), resample=Image.BILINEAR)
    if gt_img.size != (W, H):
        gt_img = gt_img.resize((W, H), resample=Image.BILINEAR)

    gap = 8
    top_bar = 0
    canvas = Image.new("RGB", (W * 3 + gap * 2, H + top_bar), (20, 20, 20))
    canvas.paste(inp_img, (0, top_bar))
    canvas.paste(out_img, (W + gap, top_bar))
    canvas.paste(gt_img, (2 * (W + gap), top_bar))

    font = get_font(22)

    # Input panel block
    ip_psnr, ip_ssim = inp_metrics
    draw_label_block(
        canvas,
        x0=12,
        y0=12,
        title="INPUT",
        lines=[f"PSNR(vs GT): {ip_psnr:.2f}", f"SSIM(vs GT): {ip_ssim:.4f}"],
        font=font
    )

    # Restored panel block
    op_psnr, op_ssim = out_metrics
    draw_label_block(
        canvas,
        x0=W + gap + 12,
        y0=12,
        title="RESTORED",
        lines=[f"PSNR(vs GT): {op_psnr:.2f}", f"SSIM(vs GT): {op_ssim:.4f}"],
        font=font
    )

    # GT panel label (no metrics)
    draw_label_block(
        canvas,
        x0=2 * (W + gap) + 12,
        y0=12,
        title="GT",
        lines=[],
        font=font
    )

    ensure_dir(os.path.dirname(save_path))
    canvas.save(save_path, quality=95)


# -------------------------
# Inference
# -------------------------
@torch.no_grad()
def run_one(
    model: torch.nn.Module,
    device: torch.device,
    input_path: str,
    gt_path: str,
    out_path: str,
    use_amp: bool = True
):
    inp_pil = load_image_rgb(input_path)
    gt_pil = load_image_rgb(gt_path)

    x = pil_to_tensor_01(inp_pil).to(device)
    # forward
    if use_amp and device.type == "cuda":
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            y = model(x)
    else:
        y = model(x)

    out_pil = tensor_01_to_pil(y)

    # metrics computed on aligned size
    W, H = inp_pil.size
    gt_aligned = gt_pil.resize((W, H), resample=Image.BILINEAR)

    inp_np = np.array(inp_pil).astype(np.float32) / 255.0
    out_np = np.array(out_pil).astype(np.float32) / 255.0
    gt_np = np.array(gt_aligned).astype(np.float32) / 255.0

    inp_m = psnr_ssim(gt_np, inp_np)
    out_m = psnr_ssim(gt_np, out_np)

    make_triplet_canvas(inp_pil, out_pil, gt_aligned, inp_m, out_m, out_path)
    return inp_m, out_m


def list_images(d: str) -> List[str]:
    exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff", "*.webp")
    files = []
    for e in exts:
        files.extend(glob.glob(os.path.join(d, e)))
    files = sorted(files)
    return files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True, help="path to .pth (state_dict)")
    ap.add_argument("--outdir", type=str, required=True, help="output directory for triplet images")
    ap.add_argument("--use_amp", type=int, default=1, help="1: AMP on (cuda only), 0: off")

    # mode A: single pair
    ap.add_argument("--input", type=str, default=None, help="single input image path")
    ap.add_argument("--gt", type=str, default=None, help="single gt image path")

    # mode B: folders (same filename matching)
    ap.add_argument("--input_dir", type=str, default=None, help="directory of inputs")
    ap.add_argument("--gt_dir", type=str, default=None, help="directory of GTs (same basenames)")

    # model args (must match training)
    ap.add_argument("--dim", type=int, default=48)
    ap.add_argument("--bias", type=int, default=0)
    ap.add_argument("--volterra_rank", type=int, default=4)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[Device]", device)

    model = RestormerVolterra(dim=args.dim, bias=bool(args.bias), volterra_rank=args.volterra_rank)
    ckpt = torch.load(args.ckpt, map_location="cpu")
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]
    missing, unexpected = model.load_state_dict(ckpt, strict=False)
    print(f"[CKPT] loaded: {args.ckpt}")
    print(f"  missing: {len(missing)} unexpected: {len(unexpected)}")

    model.to(device).eval()
    ensure_dir(args.outdir)

    use_amp = bool(args.use_amp)

    # decide mode
    if args.input is not None and args.gt is not None:
        base = os.path.splitext(os.path.basename(args.input))[0]
        out_path = os.path.join(args.outdir, f"{base}__triplet.png")
        inp_m, out_m = run_one(model, device, args.input, args.gt, out_path, use_amp=use_amp)
        print("[Saved]", out_path)
        print(f"[Metrics] INPUT  PSNR={inp_m[0]:.2f} SSIM={inp_m[1]:.4f}")
        print(f"[Metrics] OUTPUT PSNR={out_m[0]:.2f} SSIM={out_m[1]:.4f}")
        return

    if args.input_dir is not None and args.gt_dir is not None:
        inp_files = list_images(args.input_dir)
        if len(inp_files) == 0:
            raise RuntimeError(f"No images found in input_dir: {args.input_dir}")

        # batch over files, match by basename
        ok = 0
        for ip in inp_files:
            name = os.path.basename(ip)
            gp = os.path.join(args.gt_dir, name)
            if not os.path.exists(gp):
                # try same stem with any extension
                stem = os.path.splitext(name)[0]
                cand = []
                for ext in [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"]:
                    p2 = os.path.join(args.gt_dir, stem + ext)
                    if os.path.exists(p2):
                        cand.append(p2)
                if len(cand) == 0:
                    print("[Skip] GT not found for:", name)
                    continue
                gp = cand[0]

            stem = os.path.splitext(name)[0]
            out_path = os.path.join(args.outdir, f"{stem}__triplet.png")
            inp_m, out_m = run_one(model, device, ip, gp, out_path, use_amp=use_amp)
            ok += 1
            print(f"[{ok}] {name} | IN: {inp_m[0]:.2f}/{inp_m[1]:.4f}  OUT: {out_m[0]:.2f}/{out_m[1]:.4f}")

        print(f"[Done] saved {ok} triplets to: {args.outdir}")
        return

    raise RuntimeError("Provide either (--input, --gt) or (--input_dir, --gt_dir).")


if __name__ == "__main__":
    main()


"""
python "E:\restormer+volterra\Restormer + Volterra\infer_triplet_with_metrics.py" `
  --ckpt "E:\restormer+volterra\checkpoints\#01_all_tasks_balanced_160\epoch_100_ssim0.9177_psnr32.58.pth" `
  --input "E:\restormer+volterra\data\CSD\Test\Snow\111.tif" `
  --gt    "E:\restormer+volterra\data\CSD\Test\Gt\111.tif" `
  --outdir "E:\restormer+volterra\results\triplets" `
  --dim 48 --bias 0 --volterra_rank 4 --use_amp 1

"""