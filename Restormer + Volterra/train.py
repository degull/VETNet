# train.py
# E:\restormer+volterra\Restormer + Volterra\train.py
""" import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import transforms
from tqdm import tqdm
from torch.amp import autocast, GradScaler  # ✅ 최신 버전 사용
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim

from models.restormer_volterra import RestormerVolterra
from re_dataset.rain100h_dataset import Rain100HDataset
from re_dataset.gopro_dataset import GoProDataset
from re_dataset.sidd_dataset import SIDD_Dataset

# ✅ 학습 설정
BATCH_SIZE = 2
EPOCHS = 100
LR = 2e-4
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ✅ 경로 설정
RAIN100H_DIR = 'E:/restormer+volterra/data/rain100H/train'
GOPRO_CSV = 'E:/restormer+volterra/data/GOPRO_Large/gopro_train_pairs.csv'
SIDD_DIR = 'E:/restormer+volterra/data/SIDD'

SAVE_DIR = 'checkpoints/restormer_volterra_train_4sets'
os.makedirs(SAVE_DIR, exist_ok=True)

# ✅ Progressive Learning 스케줄
resize_schedule = {
    0: 128,
    30: 192,
    60: 256
}

def get_transform(epoch):
    size = 256
    for key in sorted(resize_schedule.keys()):
        if epoch >= key:
            size = resize_schedule[key]
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor()
    ])

def main():
    model = RestormerVolterra().to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scaler = GradScaler(device='cuda')

    for epoch in range(EPOCHS):
        transform = get_transform(epoch)

        # ✅ 데이터셋 로드
        rain100h_dataset = Rain100HDataset(root_dir=RAIN100H_DIR, transform=transform)
        gopro_dataset = GoProDataset(csv_path=GOPRO_CSV, transform=transform)
        sidd_dataset = SIDD_Dataset(root_dir=SIDD_DIR, transform=transform)

        # ✅ 데이터셋 통합
        train_dataset = ConcatDataset([
            rain100h_dataset,
            gopro_dataset,
            sidd_dataset
        ])

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                                  shuffle=True, num_workers=4, pin_memory=True)

        print(f"[Epoch {epoch+1}] Input resolution: {transform.transforms[0].size}, Total samples: {len(train_dataset)}")

        model.train()
        epoch_loss = 0
        total_psnr, total_ssim, count = 0.0, 0.0, 0
        loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{EPOCHS}]", leave=False)

        for distorted, reference in loop:
            distorted, reference = distorted.to(DEVICE), reference.to(DEVICE)
            optimizer.zero_grad()

            with autocast(device_type='cuda'):
                output = model(distorted)
                loss = criterion(output, reference)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()

            # ✅ PSNR / SSIM 계산 (batch 내 첫 샘플 기준)
            out_np = output[0].detach().cpu().numpy().transpose(1, 2, 0)
            ref_np = reference[0].detach().cpu().numpy().transpose(1, 2, 0)

            psnr = compute_psnr(ref_np, out_np, data_range=1.0)
            ssim = compute_ssim(ref_np, out_np, channel_axis=2, data_range=1.0, win_size=7)

            total_psnr += psnr
            total_ssim += ssim
            count += 1

            loop.set_postfix(loss=loss.item(), psnr=f"{psnr:.2f}", ssim=f"{ssim:.3f}")

        print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {epoch_loss / len(train_loader):.6f} | "
              f"Avg PSNR: {total_psnr / count:.2f} | Avg SSIM: {total_ssim / count:.4f}")

        # ✅ 모델 저장
        torch.save(model.state_dict(), os.path.join(SAVE_DIR, f"epoch_{epoch+1}.pth"))

if __name__ == '__main__':
    main()  """





# 이어서 학습
# train.py
# E:/MRVNet2D/Restormer + Volterra/train.py
""" 
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import transforms
from tqdm import tqdm
from torch.amp import autocast, GradScaler
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim

from restormer_volterra import RestormerVolterra
from kadid_dataset import KADID10KDataset
from re_dataset.rain100h_dataset import Rain100HDataset
from re_dataset.gopro_dataset import GoProDataset
from re_dataset.sidd_dataset import SIDD_Dataset

# ✅ 설정
BATCH_SIZE = 2
EPOCHS = 100
LR = 2e-4
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ✅ 경로
KADID_CSV = 'E:/restormer+volterra/data/KADID10K/kadid10k.csv'
RAIN100H_DIR = 'E:/restormer+volterra/data/rain100H/train'
GOPRO_CSV = 'E:/restormer+volterra/data/GOPRO_Large/gopro_train_pairs.csv'
SIDD_DIR = 'E:/restormer+volterra/data/SIDD'
SAVE_DIR = 'checkpoints/restormer_volterra_train_4sets'
CHECKPOINT_PATH = os.path.join(SAVE_DIR, 'epoch_98.pth')
os.makedirs(SAVE_DIR, exist_ok=True)

# ✅ Progressive Learning
resize_schedule = {
    0: 128,
    30: 192,
    60: 256
}

def get_transform(epoch):
    size = 256
    for key in sorted(resize_schedule.keys()):
        if epoch >= key:
            size = resize_schedule[key]
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor()
    ])

def main():
    model = RestormerVolterra().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scaler = GradScaler(device='cuda')

    resume_epoch = 0

    # ✅ 체크포인트가 존재하면 이어서 학습
    if os.path.exists(CHECKPOINT_PATH):
        print(f"🔁 Loading checkpoint from {CHECKPOINT_PATH}")
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
        model.load_state_dict(checkpoint)
        resume_epoch = 98  # 수동 설정 (파일명 기준)

    criterion = nn.MSELoss()

    for epoch in range(resume_epoch, EPOCHS):
        transform = get_transform(epoch)

        # ✅ 데이터셋
        kadid_dataset = KADID10KDataset(csv_file=KADID_CSV, transform=transform)
        rain100h_dataset = Rain100HDataset(root_dir=RAIN100H_DIR, transform=transform)
        gopro_dataset = GoProDataset(csv_path=GOPRO_CSV, transform=transform)
        sidd_dataset = SIDD_Dataset(root_dir=SIDD_DIR, transform=transform)

        train_dataset = ConcatDataset([kadid_dataset, rain100h_dataset, gopro_dataset, sidd_dataset])
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

        print(f"[Epoch {epoch+1}] Input resolution: {transform.transforms[0].size}, Total samples: {len(train_dataset)}")

        model.train()
        epoch_loss = 0
        total_psnr, total_ssim, count = 0.0, 0.0, 0
        loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{EPOCHS}]", leave=False)

        for distorted, reference in loop:
            distorted, reference = distorted.to(DEVICE), reference.to(DEVICE)
            optimizer.zero_grad()

            with autocast(device_type='cuda'):
                output = model(distorted)
                loss = criterion(output, reference)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()

            # PSNR/SSIM (batch 첫 샘플 기준)
            out_np = output[0].detach().cpu().numpy().transpose(1, 2, 0)
            ref_np = reference[0].detach().cpu().numpy().transpose(1, 2, 0)

            psnr = compute_psnr(ref_np, out_np, data_range=1.0)
            ssim = compute_ssim(ref_np, out_np, channel_axis=2, data_range=1.0, win_size=7)

            total_psnr += psnr
            total_ssim += ssim
            count += 1

            loop.set_postfix(loss=loss.item(), psnr=f"{psnr:.2f}", ssim=f"{ssim:.3f}")

        print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {epoch_loss / len(train_loader):.6f} | "
              f"Avg PSNR: {total_psnr / count:.2f} | Avg SSIM: {total_ssim / count:.4f}")

        # ✅ 저장
        torch.save(model.state_dict(), os.path.join(SAVE_DIR, f"epoch_{epoch+1}.pth"))

if __name__ == '__main__':
    main()
 """
# 🏆 Best Epoch: 97 | PSNR: 28.76 | SSIM: 0.8687

# E:\restormer+volterra\Restormer + Volterra\train_reside.py
# RESIDE-6K (Haze) 단독 학습 + 결과 시각화 저장
# - train/test split 사용
# - AMP + GradScaler
# - epoch마다 test 평균 PSNR/SSIM
# - results 폴더에 (원본|복원|GT) 가로로 붙인 PNG 저장
# - PNG 아래에 PSNR/SSIM 텍스트도 같이 렌더링

import os
import sys
import time

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
# Path setup (models: current dir, re_dataset: repo root)
# ----------------------
CUR_DIR = os.path.dirname(os.path.abspath(__file__))  # .../Restormer + Volterra
ROOT_DIR = os.path.dirname(CUR_DIR)                   # .../restormer+volterra
for p in [CUR_DIR, ROOT_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from models.restormer_volterra import RestormerVolterra
from re_dataset.reside_dataset import RESIDEDataset


# ======================
# Config
# ======================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

RESIDE_ROOT = r"E:/restormer+volterra/data/RESIDE-6K"

SAVE_DIR = r"E:/restormer+volterra/checkpoints/restormer_volterra_reside"
os.makedirs(SAVE_DIR, exist_ok=True)

RESULTS_DIR = r"E:/restormer+volterra/results/reside_train"
os.makedirs(RESULTS_DIR, exist_ok=True)

BATCH_SIZE = 1               # haze는 해상도 커서 1 추천 (OOM 방지)
EPOCHS = 100
LR = 2e-4
NUM_WORKERS = 4
PIN_MEMORY = True

USE_AMP = True               # AMP 켜기
SAVE_VIS_EVERY_EPOCH = 1     # 매 epoch마다 시각화 저장 (원하면 5,10 등으로)

# 각 epoch에서 몇 장 저장할지
NUM_VIS_SAVE = 5             # test에서 앞쪽 몇 장 저장

# Progressive resize schedule
resize_schedule = {
    0: 128,
    30: 192,
    60: 256,
}


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
    """(1,C,H,W) or (C,H,W) -> HWC float32 [0,1]"""
    if x.dim() == 4:
        x = x[0]
    x = x.detach().float().clamp(0, 1).cpu().numpy()
    x = np.transpose(x, (1, 2, 0))
    return x


def hwc01_to_pil(img: np.ndarray) -> Image.Image:
    """HWC [0,1] float -> PIL RGB"""
    img_u8 = (np.clip(img, 0, 1) * 255.0).astype(np.uint8)
    return Image.fromarray(img_u8, mode="RGB")


def render_triplet_with_text(hazy_np, restored_np, gt_np, psnr, ssim,
                             title_left="Input(Hazy)", title_mid="Restored", title_right="GT",
                             pad=12, text_h=54) -> Image.Image:
    """
    가로로 (hazy|restored|gt) 붙이고,
    아래에 PSNR/SSIM 텍스트 영역을 추가한 PNG 생성
    """
    hazy_p = hwc01_to_pil(hazy_np)
    res_p = hwc01_to_pil(restored_np)
    gt_p = hwc01_to_pil(gt_np)

    w, h = hazy_p.size
    canvas_w = w * 3 + pad * 4
    canvas_h = h + text_h + pad * 3

    canvas = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    # 폰트 (윈도우 기본 폰트 시도 -> 실패하면 기본)
    try:
        font = ImageFont.truetype("arial.ttf", 18)
        font_b = ImageFont.truetype("arial.ttf", 20)
    except Exception:
        font = ImageFont.load_default()
        font_b = ImageFont.load_default()

    # 상단 타이틀
    y_title = pad
    draw.text((pad + (w // 2) - 40, y_title), title_left, fill=(0, 0, 0), font=font_b)
    draw.text((pad * 2 + w + (w // 2) - 35, y_title), title_mid, fill=(0, 0, 0), font=font_b)
    draw.text((pad * 3 + w * 2 + (w // 2) - 10, y_title), title_right, fill=(0, 0, 0), font=font_b)

    # 이미지 붙이기
    y_img = pad * 2 + 18
    x1 = pad
    x2 = pad * 2 + w
    x3 = pad * 3 + w * 2

    canvas.paste(hazy_p, (x1, y_img))
    canvas.paste(res_p, (x2, y_img))
    canvas.paste(gt_p, (x3, y_img))

    # 아래 metric 텍스트
    y_text = y_img + h + pad
    metric_text = f"PSNR: {psnr:.2f} dB    SSIM: {ssim:.4f}"
    draw.text((pad, y_text), metric_text, fill=(0, 0, 0), font=font_b)

    return canvas


def evaluate_one_epoch(model, loader, criterion, epoch: int, vis_size: int,
                       save_visuals: bool = True, num_save: int = 5):
    """
    Test 검증:
    - 평균 loss/psnr/ssim 산출
    - (옵션) 앞 num_save장에 대해 triplet 시각화 PNG 저장
    """
    model.eval()
    total_loss = 0.0
    total_psnr, total_ssim, count = 0.0, 0.0, 0

    saved = 0
    epoch_vis_dir = os.path.join(RESULTS_DIR, f"epoch_{epoch:03d}")
    if save_visuals:
        os.makedirs(epoch_vis_dir, exist_ok=True)

    with torch.no_grad():
        for i, (hazy, gt) in enumerate(tqdm(loader, desc="Validating", leave=False), start=1):
            hazy = hazy.to(DEVICE, non_blocking=True)
            gt = gt.to(DEVICE, non_blocking=True)

            with autocast(device_type="cuda", enabled=(USE_AMP and DEVICE.type == "cuda")):
                out = model(hazy)
                loss = criterion(out, gt)

            total_loss += float(loss.item())

            hazy_np = tensor_to_hwc01(hazy)
            out_np = tensor_to_hwc01(out)
            gt_np = tensor_to_hwc01(gt)

            psnr = compute_psnr(gt_np, out_np, data_range=1.0)
            ssim = compute_ssim(gt_np, out_np, channel_axis=2, data_range=1.0, win_size=7)

            total_psnr += psnr
            total_ssim += ssim
            count += 1

            # 시각화 저장
            if save_visuals and saved < num_save:
                vis = render_triplet_with_text(
                    hazy_np=hazy_np,
                    restored_np=out_np,
                    gt_np=gt_np,
                    psnr=psnr,
                    ssim=ssim
                )
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
    scaler = GradScaler(device="cuda")  # torch>=2.x

    for epoch in range(EPOCHS):
        transform, size = get_transform(epoch)

        # ----------------------
        # Dataset / Loader
        # ----------------------
        train_set = RESIDEDataset(root_dir=RESIDE_ROOT, split="train", transform=transform, strict=True)
        test_set  = RESIDEDataset(root_dir=RESIDE_ROOT, split="test",  transform=transform, strict=True)

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

        print(f"\n[Epoch {epoch+1}/{EPOCHS}] RESIDE input: {size}x{size} | train={len(train_set)} | test={len(test_set)}")

        # ----------------------
        # Train
        # ----------------------
        model.train()
        epoch_loss = 0.0
        total_psnr, total_ssim, count = 0.0, 0.0, 0

        loop = tqdm(train_loader, desc=f"Train [{epoch+1}/{EPOCHS}]", leave=False)
        t0 = time.time()

        for step, (hazy, gt) in enumerate(loop, start=1):
            hazy = hazy.to(DEVICE, non_blocking=True)
            gt = gt.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast(device_type="cuda", enabled=(USE_AMP and DEVICE.type == "cuda")):
                out = model(hazy)
                loss = criterion(out, gt)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += float(loss.item())

            # --- metric (batch 첫 샘플)
            out_np = tensor_to_hwc01(out)
            gt_np = tensor_to_hwc01(gt)

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

        # ----------------------
        # Validate (test split) + save visuals
        # ----------------------
        save_visuals = (SAVE_VIS_EVERY_EPOCH > 0) and ((epoch + 1) % SAVE_VIS_EVERY_EPOCH == 0)
        val_loss, val_psnr, val_ssim = evaluate_one_epoch(
            model=model,
            loader=test_loader,
            criterion=criterion,
            epoch=epoch + 1,
            vis_size=size,
            save_visuals=save_visuals,
            num_save=NUM_VIS_SAVE
        )
        print(f"[Test ] loss={val_loss:.6f} | psnr={val_psnr:.2f} | ssim={val_ssim:.4f}")
        if save_visuals:
            print(f"[Vis  ] Saved {NUM_VIS_SAVE} triplets to: {os.path.join(RESULTS_DIR, f'epoch_{epoch+1:03d}')}")

        # ----------------------
        # Save checkpoint (파일명에 metric 포함)
        # ----------------------
        ckpt_name = f"epoch_{epoch+1:03d}_ssim{val_ssim:.4f}_psnr{val_psnr:.2f}.pth"
        ckpt_path = os.path.join(SAVE_DIR, ckpt_name)
        torch.save(model.state_dict(), ckpt_path)
        print(f"[Save] {ckpt_path}")


if __name__ == "__main__":
    main()
