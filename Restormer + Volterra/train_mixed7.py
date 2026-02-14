# E:\restormer+volterra\Restormer + Volterra\train_mixed7.py
# Mixed-7 synthetic distortion training (KADIS-style: up to 7 distortions on-the-fly)
# - Input: clean GT images only (CLEAN/train, CLEAN/test)
# - Output: model learns to restore to GT
# - Features: AMP + GradScaler, Progressive resize, per-step loss/psnr/ssim tqdm,
#             per-epoch validation PSNR/SSIM, save triplet PNGs (input|restored|gt + metrics),
#             save checkpoint each epoch as epoch_xxx_ssimXXXX_psnrXX.XX.pth

import os
import sys
import time
import re

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from torch.amp import autocast, GradScaler
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim

from PIL import Image, ImageDraw, ImageFont

# ----------------------
# ✅ Path setup (models: current dir, re_dataset: repo root)
# ----------------------
CUR_DIR = os.path.dirname(os.path.abspath(__file__))  # .../Restormer + Volterra
ROOT_DIR = os.path.dirname(CUR_DIR)                   # .../restormer+volterra
for p in [CUR_DIR, ROOT_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from models.restormer_volterra import RestormerVolterra
from re_dataset.mixed7_synth_dataset import Mixed7SynthDataset

# ======================
# Config
# ======================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLEAN_TRAIN_DIR = r"E:/restormer+volterra/data/CLEAN/train"
CLEAN_TEST_DIR  = r"E:/restormer+volterra/data/CLEAN/test"

SAVE_DIR = r"E:/restormer+volterra/checkpoints/restormer_volterra_mixed7"
os.makedirs(SAVE_DIR, exist_ok=True)

RESULTS_DIR = r"E:/restormer+volterra/results/mixed7_train"
os.makedirs(RESULTS_DIR, exist_ok=True)

RESUME = False
RESUME_CKPT = ""

BATCH_SIZE = 1
EPOCHS = 100
LR = 2e-4
NUM_WORKERS = 4
PIN_MEMORY = True

USE_AMP = True
SAVE_VIS_EVERY_EPOCH = 1
NUM_VIS_SAVE = 5  # epoch마다 저장할 샘플 수 (train/test 각각)

resize_schedule = {0: 128, 30: 192, 60: 256}
BASE_SEED = 1234


def parse_epoch_from_ckpt(path: str) -> int:
    name = os.path.basename(path)
    m = re.search(r"epoch_(\d+)", name)
    return int(m.group(1)) if m else 0


def get_transform(epoch: int):
    size = 256
    for key in sorted(resize_schedule.keys()):
        if epoch >= key:
            size = resize_schedule[key]
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
    ]), size


def tensor_to_hwc01(x: torch.Tensor) -> np.ndarray:
    if x.dim() == 4:
        x = x[0]
    x = x.detach().float().clamp(0, 1).cpu().numpy()
    return np.transpose(x, (1, 2, 0))


def hwc01_to_pil(img: np.ndarray) -> Image.Image:
    img_u8 = (np.clip(img, 0, 1) * 255.0).astype(np.uint8)
    return Image.fromarray(img_u8, mode="RGB")


def render_triplet_with_text(inp_np, restored_np, gt_np, psnr, ssim,
                             title_left="Input(Mixed)", title_mid="Restored", title_right="GT",
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
    draw.text((pad + (w // 2) - 60, y_title), title_left, fill=(0, 0, 0), font=font_b)
    draw.text((pad * 2 + w + (w // 2) - 45, y_title), title_mid, fill=(0, 0, 0), font=font_b)
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


def evaluate_one_epoch(model, loader, criterion, epoch: int, split_name: str,
                       save_visuals: bool = True, num_save: int = 5):
    """
    split_name: "train" or "test" for saving folder
    """
    model.eval()
    total_loss = 0.0
    total_psnr, total_ssim, count = 0.0, 0.0, 0

    saved = 0
    epoch_vis_dir = os.path.join(RESULTS_DIR, f"epoch_{epoch:03d}", split_name)
    if save_visuals:
        os.makedirs(epoch_vis_dir, exist_ok=True)

    with torch.no_grad():
        for _, (inp, gt) in enumerate(tqdm(loader, desc=f"Visual {split_name}", leave=False), start=1):
            inp = inp.to(DEVICE, non_blocking=True)
            gt = gt.to(DEVICE, non_blocking=True)

            with autocast(device_type="cuda", enabled=(USE_AMP and DEVICE.type == "cuda")):
                out = model(inp)
                loss = criterion(out, gt)

            total_loss += float(loss.item())

            inp_np = tensor_to_hwc01(inp)
            out_np = tensor_to_hwc01(out)
            gt_np  = tensor_to_hwc01(gt)

            psnr = compute_psnr(gt_np, out_np, data_range=1.0)
            ssim = compute_ssim(gt_np, out_np, channel_axis=2, data_range=1.0, win_size=7)

            total_psnr += psnr
            total_ssim += ssim
            count += 1

            if save_visuals and saved < num_save:
                vis = render_triplet_with_text(inp_np, out_np, gt_np, psnr, ssim)
                out_path = os.path.join(epoch_vis_dir, f"sample_{saved+1:02d}_psnr{psnr:.2f}_ssim{ssim:.4f}.png")
                vis.save(out_path)
                saved += 1

    avg_loss = total_loss / max(len(loader), 1)
    avg_psnr = total_psnr / max(count, 1)
    avg_ssim = total_ssim / max(count, 1)
    return avg_loss, avg_psnr, avg_ssim


def main():
    torch.backends.cudnn.benchmark = True
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    model = RestormerVolterra().to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scaler = GradScaler(device="cuda")

    # Resume
    start_epoch = 0
    if RESUME:
        if not os.path.exists(RESUME_CKPT):
            raise FileNotFoundError(f"Resume checkpoint not found: {RESUME_CKPT}")
        print(f"[Resume] Loading: {RESUME_CKPT}")
        state = torch.load(RESUME_CKPT, map_location=DEVICE)
        model.load_state_dict(state, strict=True)
        last_epoch = parse_epoch_from_ckpt(RESUME_CKPT)
        start_epoch = last_epoch
        print(f"[Resume] Detected last_epoch={last_epoch} -> start from epoch {start_epoch+1}")

    for epoch_idx in range(start_epoch, EPOCHS):
        epoch_num = epoch_idx + 1
        transform, size = get_transform(epoch_num)

        train_set = Mixed7SynthDataset(
            clean_root=CLEAN_TRAIN_DIR,
            mode="train",
            transform=transform,
            seed=BASE_SEED + epoch_num,
            min_ops=1,
            max_ops=7,
        )
        test_set = Mixed7SynthDataset(
            clean_root=CLEAN_TEST_DIR,
            mode="test",
            transform=transform,
            seed=BASE_SEED,
            min_ops=1,
            max_ops=7,
        )

        train_loader = DataLoader(
            train_set,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKERS,
            pin_memory=(PIN_MEMORY and DEVICE.type == "cuda"),
            drop_last=True,
        )
        test_loader = DataLoader(
            test_set,
            batch_size=1,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=(PIN_MEMORY and DEVICE.type == "cuda"),
        )

        print(f"\n[Epoch {epoch_num}/{EPOCHS}] Mixed7 input: {size}x{size} | train={len(train_set)} | test={len(test_set)}")

        # ---- Train ----
        model.train()
        epoch_loss = 0.0
        total_psnr, total_ssim, count = 0.0, 0.0, 0
        loop = tqdm(train_loader, desc=f"Train [{epoch_num}/{EPOCHS}]", leave=False)
        t0 = time.time()

        for _, (inp, gt) in enumerate(loop, start=1):
            inp = inp.to(DEVICE, non_blocking=True)
            gt  = gt.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast(device_type="cuda", enabled=(USE_AMP and DEVICE.type == "cuda")):
                out = model(inp)
                loss = criterion(out, gt)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += float(loss.item())

            out_np = tensor_to_hwc01(out)
            gt_np  = tensor_to_hwc01(gt)
            psnr = compute_psnr(gt_np, out_np, data_range=1.0)
            ssim = compute_ssim(gt_np, out_np, channel_axis=2, data_range=1.0, win_size=7)

            total_psnr += psnr
            total_ssim += ssim
            count += 1

            loop.set_postfix(loss=f"{loss.item():.4f}", psnr=f"{psnr:.2f}", ssim=f"{ssim:.3f}")

        dt = time.time() - t0
        train_loss = epoch_loss / max(len(train_loader), 1)
        train_psnr = total_psnr / max(count, 1)
        train_ssim = total_ssim / max(count, 1)
        print(f"[Train] loss={train_loss:.6f} | psnr={train_psnr:.2f} | ssim={train_ssim:.4f} | time={dt/60:.1f}m")

        # ---- Save visuals (train + test) ----
        save_visuals = (SAVE_VIS_EVERY_EPOCH > 0) and (epoch_num % SAVE_VIS_EVERY_EPOCH == 0)
        if save_visuals:
            # train 시각화는 “고정된 샘플”이 아니라 epoch의 랜덤왜곡이므로,
            # DataLoader를 그대로 쓰되 앞 NUM_VIS_SAVE장만 저장하도록 evaluate 함수를 재사용.
            _ = evaluate_one_epoch(model, DataLoader(train_set, batch_size=1, shuffle=False, num_workers=0),
                                   criterion, epoch_num, split_name="train", save_visuals=True, num_save=NUM_VIS_SAVE)

        val_loss, val_psnr, val_ssim = evaluate_one_epoch(
            model=model,
            loader=test_loader,
            criterion=criterion,
            epoch=epoch_num,
            split_name="test",
            save_visuals=save_visuals,
            num_save=NUM_VIS_SAVE
        )
        print(f"[Test ] loss={val_loss:.6f} | psnr={val_psnr:.2f} | ssim={val_ssim:.4f}")
        if save_visuals:
            print(f"[Vis  ] Saved train/test triplets to: {os.path.join(RESULTS_DIR, f'epoch_{epoch_num:03d}')}")

        # ---- Save checkpoint ----
        ckpt_name = f"epoch_{epoch_num:03d}_ssim{val_ssim:.4f}_psnr{val_psnr:.2f}.pth"
        ckpt_path = os.path.join(SAVE_DIR, ckpt_name)
        torch.save(model.state_dict(), ckpt_path)
        print(f"[Save] {ckpt_path}")


if __name__ == "__main__":
    main()
