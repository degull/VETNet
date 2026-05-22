# VTN qualitative visualization script
# ----------------------------------------------------------
# ?섏젙? ?꾨옒 CONFIG 遺遺꾨쭔 ?섎㈃ ??
# ?ㅽ뻾: 洹몃깷 python visualization.py

import os
import sys
import math

import torch
import numpy as np
from torchvision import transforms
from torch.amp import autocast

from PIL import Image, ImageDraw, ImageFont
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim


# ==========================================================
# ??CONFIG (?ш린留??섏젙?섎㈃ ??
# ==========================================================

CKPT_PATH = None

DISTORTED_IMAGE = None
GT_IMAGE        = None

OUTPUT_IMAGE = None

RESIZE_SIZE = 256      # 0?대㈃ ?먮낯 ?댁긽???좎?
USE_AMP = True
USE_TILE = False       # ???대?吏硫?True 異붿쿇
TILE_SIZE = 256
TILE_OVERLAP = 32


# ==========================================================
# Path setup
# ==========================================================
CUR_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CUR_DIR)

for p in [CUR_DIR, ROOT_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from models.restormer_volterra import RestormerVolterra
from config import DATA, CHECKPOINTS, result_dir, as_str

CKPT_PATH = as_str(CHECKPOINTS["csd"])
DISTORTED_IMAGE = as_str(DATA["csd"] / "Test" / "Snow" / "111.tif")
GT_IMAGE = as_str(DATA["csd"] / "Test" / "Gt" / "111.tif")
OUTPUT_IMAGE = as_str(result_dir("visualize_results") / "visualization_result_csd.png")


# ==========================================================
# Utility Functions
# ==========================================================

def tensor_to_hwc01(x):
    if x.dim() == 4:
        x = x[0]
    x = x.detach().float().clamp(0, 1).cpu().numpy()
    return np.transpose(x, (1, 2, 0))


def hwc01_to_pil(img):
    img_u8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    return Image.fromarray(img_u8)


def pil_to_tensor(img):
    if RESIZE_SIZE > 0:
        tfm = transforms.Compose([
            transforms.Resize((RESIZE_SIZE, RESIZE_SIZE)),
            transforms.ToTensor()
        ])
    else:
        tfm = transforms.Compose([
            transforms.ToTensor()
        ])
    return tfm(img).unsqueeze(0)


@torch.no_grad()
def tile_forward(model, x, device):
    if not USE_TILE:
        with autocast(device_type="cuda", enabled=(USE_AMP and device.type=="cuda")):
            return model(x)

    b, c, h, w = x.shape
    tile = TILE_SIZE
    overlap = TILE_OVERLAP
    stride = tile - overlap

    output = torch.zeros_like(x)
    weight = torch.zeros((1,1,h,w), device=device)

    for y in range(0, h, stride):
        for x0 in range(0, w, stride):
            y1 = min(y + tile, h)
            x1 = min(x0 + tile, w)

            patch = x[:, :, y:y1, x0:x1]
            with autocast(device_type="cuda", enabled=(USE_AMP and device.type=="cuda")):
                out_patch = model(patch)

            output[:, :, y:y1, x0:x1] += out_patch
            weight[:, :, y:y1, x0:x1] += 1

    return output / weight


def compute_metrics(gt, pred):
    psnr = compute_psnr(gt, pred, data_range=1.0)
    ssim = compute_ssim(gt, pred, data_range=1.0, channel_axis=2, win_size=7)
    return psnr, ssim


def render_triplet(gt_np, inp_np, out_np,
                   inp_psnr, inp_ssim,
                   out_psnr, out_ssim):

    gt_p = hwc01_to_pil(gt_np)
    inp_p = hwc01_to_pil(inp_np)
    out_p = hwc01_to_pil(out_np)

    w, h = gt_p.size
    pad = 12
    text_h = 100

    canvas = Image.new("RGB", (w*3 + pad*4, h + text_h + pad*3), (255,255,255))
    draw = ImageDraw.Draw(canvas)

    try:
        font_title = ImageFont.truetype("arial.ttf", 20)
        font_text  = ImageFont.truetype("arial.ttf", 18)
    except:
        font_title = ImageFont.load_default()
        font_text  = ImageFont.load_default()

    # Titles
    draw.text((pad + w//2 - 15, pad), "GT", font=font_title, fill=(0,0,0))
    draw.text((pad*2 + w + w//2 - 45, pad), "Distorted", font=font_title, fill=(0,0,0))
    draw.text((pad*3 + w*2 + w//2 - 40, pad), "Restored", font=font_title, fill=(0,0,0))

    y_img = pad*2 + 25
    canvas.paste(gt_p,  (pad, y_img))
    canvas.paste(inp_p, (pad*2 + w, y_img))
    canvas.paste(out_p, (pad*3 + w*2, y_img))

    y_text = y_img + h + pad
    draw.text((pad, y_text),
              f"Distorted vs GT  : PSNR {inp_psnr:.2f} dB | SSIM {inp_ssim:.4f}",
              font=font_text, fill=(0,0,0))

    draw.text((pad, y_text + 35),
              f"Restored  vs GT  : PSNR {out_psnr:.2f} dB | SSIM {out_ssim:.4f}",
              font=font_text, fill=(0,0,0))

    return canvas


# ==========================================================
# Main
# ==========================================================

def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(CKPT_PATH):
        raise FileNotFoundError(CKPT_PATH)

    os.makedirs(os.path.dirname(OUTPUT_IMAGE), exist_ok=True)

    # Load images
    inp_pil = Image.open(DISTORTED_IMAGE).convert("RGB")
    gt_pil  = Image.open(GT_IMAGE).convert("RGB")

    inp = pil_to_tensor(inp_pil).to(device)
    gt  = pil_to_tensor(gt_pil).to(device)

    # Load model
    model = RestormerVolterra().to(device)
    state = torch.load(CKPT_PATH, map_location=device)
    model.load_state_dict(state, strict=True)
    model.eval()

    # Forward
    with torch.no_grad():
        out = tile_forward(model, inp, device)

    inp_np = tensor_to_hwc01(inp)
    gt_np  = tensor_to_hwc01(gt)
    out_np = tensor_to_hwc01(out)

    inp_psnr, inp_ssim = compute_metrics(gt_np, inp_np)
    out_psnr, out_ssim = compute_metrics(gt_np, out_np)

    vis = render_triplet(gt_np, inp_np, out_np,
                         inp_psnr, inp_ssim,
                         out_psnr, out_ssim)

    vis.save(OUTPUT_IMAGE)

    print("Saved:", OUTPUT_IMAGE)
    print("Distorted  PSNR:", inp_psnr, "SSIM:", inp_ssim)
    print("Restored   PSNR:", out_psnr, "SSIM:", out_ssim)


if __name__ == "__main__":
    main()

