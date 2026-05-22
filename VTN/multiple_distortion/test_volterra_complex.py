import os, sys, torch
sys.path.append(r"E:/restormer+volterra/Restormer + Volterra/")  # ✅ 프로젝트 경로 추가

import numpy as np
from PIL import Image
from torchvision import transforms
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim
from tqdm import tqdm

# ✅ Restormer+Volterra 모델

from models.restormer_volterra import RestormerVolterra
# ✅ ARNIQA 복합왜곡 생성
from pipeline import apply_random_distortions

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ----------------- Config -----------------
CKPTS = {
    "Volterra-derain-h128": r"E:\restormer+volterra\Restormer + Volterra\tasks\deraining\checkpoint_rain100h\epoch_128_ssim0.9862_psnr41.30.pth",
    "Volterra-derain-h98":  r"E:\restormer+volterra\Restormer + Volterra\tasks\deraining\checkpoint_rain100hs\epoch_98_ssim0.9786_psnr38.77.pth",
    "Volterra-desnow":      r"E:\restormer+volterra\checkpoints\restormer_volterra_csd\epoch_5_ssim0.9531_psnr33.03.pth",
    "Volterra-denoise":     r"E:\restormer+volterra\checkpoints\restormer_volterra_sidd\epoch_100.pth",
    # "Volterra-deblur":   "pth 없음 → 제외"
}

GT_DIRS = {
    "Volterra-derain-h128": r"E:\restormmer\Restormer\data\rain100H\test\norain",
    "Volterra-derain-h98":  r"E:\restormmer\Restormer\data\rain100H\test\norain",
    "Volterra-desnow":      r"E:\restormmer\Restormer\data\CSD\Test\Gt",
    "Volterra-denoise":     r"E:\restormmer\Restormer\data\SIDD\Data",  # GT/NOISY 쌍 존재
}

RESIZE = 256
MAX_TEST_IMG = 100   # 너무 많으면 시간 오래 걸리므로 제한
to_tensor = transforms.ToTensor()
to_pil = transforms.ToPILImage()

# ----------------- 유틸 -----------------
def load_imgs_from_dir(img_dir, resize=256, max_num=None):
    exts = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]
    img_paths = sorted([os.path.join(img_dir, f) for f in os.listdir(img_dir)
                        if os.path.splitext(f)[1].lower() in exts])

    if max_num is not None:
        img_paths = img_paths[:max_num]

    imgs = []
    for p in img_paths:
        try:
            img = Image.open(p).convert("RGB")
            if resize:
                img = img.resize((resize, resize), Image.BICUBIC)
            imgs.append(to_tensor(img).unsqueeze(0))
        except Exception as e:
            print(f"[WARNING] Skip file {p} due to error: {e}")
            continue

    print(f"[INFO] Loaded {len(imgs)} images from {img_dir}")
    return imgs

def evaluate_model(model, ckpt_path, clean_imgs):
    state = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)

    if "params" in state:
        state = state["params"]
    elif "state_dict" in state:
        state = state["state_dict"]

    model.load_state_dict(state, strict=False)
    model.eval()

    psnrs, ssims = [], []
    for idx, gt in enumerate(tqdm(clean_imgs, desc=f"Testing {os.path.basename(ckpt_path)}")):
        print(f"[DEBUG] processing image {idx+1}/{len(clean_imgs)}")
        gt = gt.to(DEVICE)

        # ✅ 복합왜곡 생성
        distorted = apply_random_distortions(to_pil(gt.squeeze().cpu()))
        distorted = to_tensor(distorted).unsqueeze(0).to(DEVICE)

        # 복원
        with torch.no_grad():
            restored = model(distorted).clamp(0, 1)

        # 지표 계산
        gt_np = gt.squeeze().cpu().permute(1, 2, 0).numpy()
        res_np = restored.squeeze().cpu().permute(1, 2, 0).numpy()
        psnrs.append(compute_psnr(gt_np, res_np, data_range=1.0))
        ssims.append(compute_ssim(gt_np, res_np, channel_axis=2, data_range=1.0))

    return np.mean(psnrs), np.mean(ssims)

# ----------------- 메인 -----------------
if __name__ == "__main__":
    for name, ckpt in CKPTS.items():
        print(f"\n==== Evaluating {name} ====")
        img_dir = GT_DIRS[name]
        clean_imgs = load_imgs_from_dir(img_dir, RESIZE, MAX_TEST_IMG)
        model = RestormerVolterra().to(DEVICE)

        if len(clean_imgs) == 0:
            print(f"[ERROR] No valid images found in {img_dir}")
            continue

        avg_psnr, avg_ssim = evaluate_model(model, ckpt, clean_imgs)
        print(f"{name}: PSNR={avg_psnr:.2f}, SSIM={avg_ssim:.4f}")
