# --------------------------------------------------------------
# ✅ Ablation Evaluation for Volterra Variants (Rain100H)
# --------------------------------------------------------------
import os, time, torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from thop import profile
from skimage.metrics import peak_signal_noise_ratio as psnr_metric
from skimage.metrics import structural_similarity as ssim_metric

# --------------------------------------------------------------
# ✅ 모델 import
# --------------------------------------------------------------
from models.restormer_volterra import RestormerVolterra

# --------------------------------------------------------------
# ✅ 데이터셋 경로 (Rain100H)
# --------------------------------------------------------------
GT_DIR = r"E:/restormer+volterra/data/rain100H/test/norain"
INPUT_DIR = r"E:/restormer+volterra/data/rain100H/test/rain"

# ✅ 공통 checkpoint 경로
CKPT_PATH = r"E:\restormer+volterra\checkpoints\restormer_volterra_rain100h\epoch_100.pth"

device = "cuda" if torch.cuda.is_available() else "cpu"
to_tensor = transforms.ToTensor()

# --------------------------------------------------------------
# ✅ PSNR / SSIM 계산 함수
# --------------------------------------------------------------
def evaluate_folder(model, input_dir, gt_dir, max_eval=20):
    psnr_total, ssim_total = 0, 0
    count = 0
    model.eval()
    with torch.no_grad():
        for i, fname in enumerate(tqdm(os.listdir(input_dir), desc="Evaluating")):
            if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
                continue
            inp = Image.open(os.path.join(input_dir, fname)).convert("RGB")
            gt = Image.open(os.path.join(gt_dir, fname)).convert("RGB")

            inp_t = to_tensor(inp).unsqueeze(0).to(device)
            gt_t = to_tensor(gt).unsqueeze(0).to(device)

            with torch.cuda.amp.autocast():
                pred = model(inp_t)
            pred = torch.clamp(pred, 0, 1)

            pred_np = pred.squeeze(0).permute(1, 2, 0).cpu().numpy()
            gt_np = gt_t.squeeze(0).permute(1, 2, 0).cpu().numpy()
            psnr_val = psnr_metric(gt_np, pred_np, data_range=1.0)
            ssim_val = ssim_metric(gt_np, pred_np, data_range=1.0, channel_axis=2)

            psnr_total += psnr_val
            ssim_total += ssim_val
            count += 1
            if count >= max_eval:
                break

    return psnr_total / count, ssim_total / count

# --------------------------------------------------------------
# ✅ Ablation 구성 (각기 다른 Volterra 활성 설정)
# --------------------------------------------------------------
configs = {
    "A_Baseline": dict(use_volterra_mdta=False, use_volterra_gdfn=False),
    "B_Vol_MDTA": dict(use_volterra_mdta=True,  use_volterra_gdfn=False),
    "C_Vol_GDFN": dict(use_volterra_mdta=False, use_volterra_gdfn=True),
    "D_Full_VET": dict(use_volterra_mdta=True,  use_volterra_gdfn=True),
}

results = {}

# --------------------------------------------------------------
# ✅ 실험 실행
# --------------------------------------------------------------
for name, cfg in configs.items():
    print(f"\n===== Evaluating {name} =====")
    model = RestormerVolterra(**cfg).to(device)

    # Load same pretrained weights
    if os.path.exists(CKPT_PATH):
        state = torch.load(CKPT_PATH, map_location=device)
        model.load_state_dict(state, strict=False)
        print(f"→ Loaded checkpoint from: {CKPT_PATH}")
    else:
        print(f"[Warning] Checkpoint not found at {CKPT_PATH}")

    model.eval()

    # FLOPs / Params
    dummy = torch.randn(1, 3, 256, 256).to(device)
    flops, _ = profile(model, inputs=(dummy,), verbose=False)
    flops_g = flops / 1e9
    params_m = sum(p.numel() for p in model.parameters()) / 1e6

    # Inference Time
    torch.cuda.synchronize()
    N = 20
    t0 = time.time()
    with torch.no_grad():
        for _ in range(N):
            _ = model(dummy)
    torch.cuda.synchronize()
    avg_time = (time.time() - t0) / N * 1000  # ms

    # VRAM 사용량
    torch.cuda.synchronize()
    mem = torch.cuda.max_memory_allocated(device) / (1024**2)
    torch.cuda.reset_peak_memory_stats()

    # PSNR / SSIM (Rain100H test)
    psnr, ssim = evaluate_folder(model, INPUT_DIR, GT_DIR, max_eval=15)

    results[name] = dict(
        Params=params_m,
        FLOPs=flops_g,
        Time=avg_time,
        VRAM=mem,
        PSNR=psnr,
        SSIM=ssim
    )

# --------------------------------------------------------------
# ✅ 결과 출력
# --------------------------------------------------------------
print("\n================ Ablation: Volterra Efficiency =================")
print(f"{'Model':<15} {'Params(M)':<10} {'FLOPs(G)':<10} {'Time(ms)':<10} {'VRAM(MB)':<10} {'PSNR':<8} {'SSIM':<8}")
for k, v in results.items():
    print(f"{k:<15} {v['Params']:<10.2f} {v['FLOPs']:<10.2f} {v['Time']:<10.2f} {v['VRAM']:<10.1f} {v['PSNR']:<8.3f} {v['SSIM']:<8.4f}")

