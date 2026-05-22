# E:/restormer+volterra/test_alltasks_complex.py
""" 
import os
import sys
import glob
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim

# 모델 import
sys.path.append(r"E:/restormer+volterra/Restormer + Volterra")
from models.restormer_volterra import RestormerVolterra

# 복합왜곡 파이프라인 (ARNIQA 스타일)
from multiple_distortion.pipeline import distort_images

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 체크포인트 경로
CKPT = r"E:/restormer+volterra/checkpoints/#01_all_tasks_balanced_160/epoch_100_ssim0.9177_psnr32.58.pth"

# GT 데이터셋 디렉토리들
CLEAN_DIRS = [
    r"E:/restormer+volterra/data/CSD/Test/Gt",
    r"E:/restormer+volterra/data/HIDE/GT",
    r"E:/restormer+volterra/data/rain100H/test/norain",
    r"E:/restormer+volterra/data/rain100L/test/norain",
    r"E:/restormer+volterra/data/SIDD/Data",
]

# 허용 확장자 (tif/tiff 추가)
ALLOWED_EXT = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]

# 저장 디렉토리
DISTORTION_DIR = r"E:/restormer+volterra/Restormer + Volterra/distortion_imgs/distortion"
RESULTS_DIR = r"E:/restormer+volterra/Restormer + Volterra/distortion_imgs/results"
os.makedirs(DISTORTION_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


# ----------------- 유틸 함수 -----------------
def load_img(path, resize=256):
    img = Image.open(path).convert("RGB")
    if resize:
        img = img.resize((resize, resize), Image.BICUBIC)
    return img


def tensor_to_numpy(t):
    arr = t.detach().cpu().numpy()
    arr = np.transpose(arr, (1, 2, 0))
    return np.clip(arr, 0, 1)


# ----------------- 메인 -----------------
def main():
    # 모델 로드
    print(f"[INFO] Loading checkpoint: {CKPT}")
    model = RestormerVolterra()
    state = torch.load(CKPT, map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE).eval()

    avg_psnr, avg_ssim = [], []
    img_counter = 0

    for clean_dir in CLEAN_DIRS:
        img_paths = sorted(glob.glob(os.path.join(clean_dir, "*.*")))
        img_paths = [p for p in img_paths if os.path.splitext(p)[1].lower() in ALLOWED_EXT]

        print(f"\n[INFO] Processing dataset: {clean_dir} | {len(img_paths)} images")

        for idx, path in enumerate(tqdm(img_paths, desc=os.path.basename(clean_dir))):
            img_counter += 1
            # GT 이미지 로드
            gt_img = load_img(path)
            gt_tensor = transforms.ToTensor()(gt_img).to(DEVICE).unsqueeze(0)

            # 복합 왜곡 생성
            distorted, funcs, vals = distort_images(gt_tensor.squeeze(0).clone(),
                                                    max_distortions=4, num_levels=5)
            distorted_tensor = distorted.unsqueeze(0).to(DEVICE)

            # 복원
            with torch.no_grad():
                restored = model(distorted_tensor)

            # numpy 변환
            restored_np = tensor_to_numpy(restored.squeeze(0))
            gt_np = tensor_to_numpy(gt_tensor.squeeze(0))
            distorted_np = tensor_to_numpy(distorted_tensor.squeeze(0))

            # PSNR / SSIM 계산
            psnr = compute_psnr(gt_np, restored_np, data_range=1.0)
            ssim = compute_ssim(gt_np, restored_np, channel_axis=2, data_range=1.0)

            avg_psnr.append(psnr)
            avg_ssim.append(ssim)

            # 저장
            base_name = f"img_{img_counter:05d}"
            distorted_path = os.path.join(DISTORTION_DIR, f"{base_name}_distorted.png")
            restored_path = os.path.join(RESULTS_DIR, f"{base_name}_restored.png")

            Image.fromarray((distorted_np * 255).astype(np.uint8)).save(distorted_path)
            Image.fromarray((restored_np * 255).astype(np.uint8)).save(restored_path)

            # 로그 출력
            print(f"[{img_counter}] {os.path.basename(path)} | "
                  f"PSNR: {psnr:.2f}, SSIM: {ssim:.4f} | "
                  f"Distortions: {[f.__name__ for f in funcs]} | Levels: {vals}")

    # 최종 평균 결과
    print(f"\n==== Final Results on Complex Distortions ({img_counter} images) ====")
    print(f"Average PSNR: {np.mean(avg_psnr):.2f}")
    print(f"Average SSIM: {np.mean(avg_ssim):.4f}")


if __name__ == "__main__":
    main()
 """

# E:/restormer+volterra/test_alltasks_complex.py
import os
import sys
import glob
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim

# 모델 import
sys.path.append(r"E:/restormer+volterra/Restormer + Volterra")
from models.restormer_volterra import RestormerVolterra

# 복합왜곡 파이프라인 (ARNIQA 스타일)
from multiple_distortion.pipeline import distort_images

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 체크포인트 경로
CKPT = r"E:/restormer+volterra/checkpoints/#01_all_tasks_balanced_160/epoch_100_ssim0.9177_psnr32.58.pth"

# GT 데이터셋 디렉토리들
CLEAN_DIRS = [
    r"E:/restormer+volterra/data/CSD/Test/Gt",
    r"E:/restormer+volterra/data/HIDE/GT",
    r"E:/restormer+volterra/data/rain100H/test/norain",
    r"E:/restormer+volterra/data/rain100L/test/norain",
    r"E:/restormer+volterra/data/SIDD/Data",
]

# 허용 확장자
ALLOWED_EXT = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]

# ----------------- 유틸 함수 -----------------
def load_img(path, resize=256):
    """이미지 로드 후 RGB 변환 및 리사이즈"""
    img = Image.open(path).convert("RGB")
    if resize:
        img = img.resize((resize, resize), Image.BICUBIC)
    return img

def tensor_to_numpy(t):
    """torch.Tensor(C,H,W) → numpy(H,W,C)"""
    arr = t.detach().cpu().numpy()
    arr = np.transpose(arr, (1, 2, 0))
    return np.clip(arr, 0, 1)

# ----------------- 메인 -----------------
def main():
    # 모델 로드
    print(f"[INFO] Loading checkpoint: {CKPT}")
    model = RestormerVolterra()
    state = torch.load(CKPT, map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE).eval()

    psnr_restored, ssim_restored = [], []
    psnr_distorted, ssim_distorted = [], []
    img_counter = 0

    for clean_dir in CLEAN_DIRS:
        img_paths = sorted(glob.glob(os.path.join(clean_dir, "*.*")))
        img_paths = [p for p in img_paths if os.path.splitext(p)[1].lower() in ALLOWED_EXT]

        print(f"[INFO] Processing dataset: {clean_dir} | {len(img_paths)} images")

        for path in tqdm(img_paths, desc=os.path.basename(clean_dir)):
            img_counter += 1
            # GT 이미지 로드
            gt_img = load_img(path)
            gt_tensor = transforms.ToTensor()(gt_img).to(DEVICE).unsqueeze(0)

            # 복합 왜곡 생성
            distorted, _, _ = distort_images(gt_tensor.squeeze(0).clone(),
                                             max_distortions=4, num_levels=5)
            distorted_tensor = distorted.unsqueeze(0).to(DEVICE)

            # 복원
            with torch.no_grad():
                restored = model(distorted_tensor)

            # numpy 변환
            restored_np = tensor_to_numpy(restored.squeeze(0))
            gt_np = tensor_to_numpy(gt_tensor.squeeze(0))
            distorted_np = tensor_to_numpy(distorted_tensor.squeeze(0))

            # PSNR / SSIM 계산
            psnr_rest = compute_psnr(gt_np, restored_np, data_range=1.0)
            ssim_rest = compute_ssim(gt_np, restored_np, channel_axis=2, data_range=1.0)
            psnr_dist = compute_psnr(gt_np, distorted_np, data_range=1.0)
            ssim_dist = compute_ssim(gt_np, distorted_np, channel_axis=2, data_range=1.0)

            psnr_restored.append(psnr_rest)
            ssim_restored.append(ssim_rest)
            psnr_distorted.append(psnr_dist)
            ssim_distorted.append(ssim_dist)

    # 최종 평균 결과
    print(f"\n==== Final Results on Complex Distortions ({img_counter} images) ====")
    print(f"Distorted → GT | PSNR={np.mean(psnr_distorted):.2f}, SSIM={np.mean(ssim_distorted):.4f}")
    print(f"Restored  → GT | PSNR={np.mean(psnr_restored):.2f}, SSIM={np.mean(ssim_restored):.4f}")

if __name__ == "__main__":
    main()
