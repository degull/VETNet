from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_ROOT = PROJECT_ROOT / "data"
CHECKPOINT_ROOT = PROJECT_ROOT / "checkpoints"
RESULTS_ROOT = PROJECT_ROOT / "results"
PAPER_ROOT = PROJECT_ROOT / "paper"

MODELS_ROOT = PROJECT_ROOT / "models"
DATASETS_ROOT = PROJECT_ROOT / "datasets"

RESULTS_ROOT.mkdir(exist_ok=True)

DATA = {
    "rain100h": DATA_ROOT / "rain100H",
    "rain100l": DATA_ROOT / "rain100L",
    "gopro": DATA_ROOT / "GOPRO_Large",
    "hide": DATA_ROOT / "HIDE",
    "reside": DATA_ROOT / "RESIDE-6K",
    "sots": DATA_ROOT / "SOTS",
    "csd": DATA_ROOT / "CSD",
    "sidd": DATA_ROOT / "SIDD",
    "kadid": DATA_ROOT / "kadid_seperate",
    "clean": DATA_ROOT / "CLEAN",
    "voc": DATA_ROOT / "VOC",
}

CHECKPOINTS = {
    "unified": CHECKPOINT_ROOT / "#01_all_tasks" / "epoch_98_ssim0.9277_psnr35.23.pth",
    "unified_alt": CHECKPOINT_ROOT / "#01_all_tasks" / "epoch_96_ssim0.9309_psnr35.19.pth",
    "unified_balanced160": CHECKPOINT_ROOT / "#01_all_tasks_balanced_160" / "epoch_99_ssim0.9183_psnr32.73.pth",
    "unified_balanced2000": CHECKPOINT_ROOT / "restormer_volterra_unified_balanced2000" / "epoch_076_ssim0.9501_psnr32.20.pth",
    "mixed7": CHECKPOINT_ROOT / "restormer_volterra_mixed7" / "epoch_048_ssim0.9119_psnr30.14.pth",
    "rain100h": CHECKPOINT_ROOT / "restormer_volterra_rain100h" / "epoch_100.pth",
    "rain100l": CHECKPOINT_ROOT / "restormer_volterra_rain100l" / "epoch_91_ssim0.9538_psnr33.98.pth",
    "gopro": CHECKPOINT_ROOT / "restormer_volterra_gopro" / "epoch_013_ssim0.9690_psnr34.80.pth",
    "reside": CHECKPOINT_ROOT / "restormer_volterra_reside" / "epoch_015_ssim0.9520_psnr27.97.pth",
    "csd": CHECKPOINT_ROOT / "restormer_volterra_csd" / "epoch_5_ssim0.9531_psnr33.03.pth",
    "sots": CHECKPOINT_ROOT / "sots_volterra" / "epoch_77_valssim0.9580_valpsnr26.83.pth",
    "sidd": CHECKPOINT_ROOT / "restormer_volterra_sidd" / "epoch_100.pth",
    "kadid_gaussian": CHECKPOINT_ROOT / "kadid_gaussian" / "epoch_89_valssim0.9455_valpsnr36.31.pth",
    "kadid_impulse": CHECKPOINT_ROOT / "kadid_impulse_noise" / "epoch_58_valssim0.9185_valpsnr35.41.pth",
    "kadid_white": CHECKPOINT_ROOT / "kadid_white_noise" / "epoch_100_valssim0.9357_valpsnr34.85.pth",
}

def result_dir(name: str) -> Path:
    path = RESULTS_ROOT / name
    path.mkdir(parents=True, exist_ok=True)
    return path

def as_str(path: Path) -> str:
    return str(path)
