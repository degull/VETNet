# 데이터셋 그대로 쓰는 generator (dataset-driven)
import os, glob
from PIL import Image

def load_img(path): 
    return Image.open(path).convert("RGB")

# ---------------- Rain100H/L ----------------
def sample_rain100_pair(base_dir, fname):
    rain_path = os.path.join(base_dir, "rain", fname)
    gt_path   = os.path.join(base_dir, "norain", fname)
    return load_img(rain_path), load_img(gt_path)

# ---------------- CSD ----------------
def sample_csd_pair(base_dir, fname):
    snow_path = os.path.join(base_dir, "Snow", fname)
    gt_path   = os.path.join(base_dir, "Gt", fname)
    return load_img(snow_path), load_img(gt_path)

# ---------------- HIDE ----------------
def sample_hide_pair(blur_path, gt_dir):
    fname = os.path.basename(blur_path)
    gt_path = os.path.join(gt_dir, fname)
    return load_img(blur_path), load_img(gt_path)

# ---------------- SIDD ----------------
def sample_sidd_pair(folder):
    noisy = load_img(os.path.join(folder, "NOISY_SRGB_010.PNG"))
    gt    = load_img(os.path.join(folder, "GT_SRGB_010.PNG"))
    return noisy, gt
