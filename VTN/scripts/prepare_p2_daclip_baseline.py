import argparse
import csv
import os
import shutil
import sys
from pathlib import Path

CUR_DIR = Path(__file__).resolve().parent
ROOT_DIR = CUR_DIR.parent
WORKSPACE_DIR = ROOT_DIR.parent
DA_CLIP_ROOT = WORKSPACE_DIR / "baselines" / "DA-CLIP"
DA_CLIP_CFG = DA_CLIP_ROOT / "universal-image-restoration" / "config" / "daclip-sde"

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import DATA


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
DISTORTIONS = ["rain100h", "rain100l", "motion-blurry", "noisy", "snowy"]


def list_images(path: Path):
    return sorted([p for p in path.iterdir() if p.suffix.lower() in IMAGE_EXTS], key=lambda x: x.name.lower())


def safe_link_or_copy(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def write_pair(src_lq: Path, src_gt: Path, lq_dir: Path, gt_dir: Path, prefix: str, idx: int):
    suffix = src_lq.suffix.lower() if src_lq.suffix else ".png"
    name = f"{prefix}_{idx:05d}{suffix}"
    safe_link_or_copy(src_lq, lq_dir / name)
    safe_link_or_copy(src_gt, gt_dir / name)


def read_csv_pairs(csv_path: Path):
    pairs = []
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if not row or len(row) < 2:
                continue
            if i == 0 and ("dist" in row[0].lower() or "blur" in row[0].lower()):
                continue
            lq = resolve_existing_path(row[0].strip())
            gt = resolve_existing_path(row[1].strip())
            if lq.exists() and gt.exists():
                pairs.append((lq, gt))
            else:
                raise FileNotFoundError(f"Missing pair from {csv_path}:\n  LQ={lq}\n  GT={gt}")
    return pairs


def resolve_existing_path(raw: str):
    p = Path(raw.strip().strip('"').strip("'").replace("\\", "/"))
    if p.exists():
        return p

    raw_norm = str(p).replace("\\", "/")
    if "/Data/" in raw_norm:
        tail = raw_norm.split("/Data/", 1)[1]
        candidate = DATA["sidd"] / "Data" / Path(tail)
        if candidate.exists():
            return candidate
    return p


def collect_rain(root: Path, max_items: int):
    lq = list_images(root / "train" / "rain")
    gt = list_images(root / "train" / "norain")
    pairs = list(zip(lq, gt))
    return pairs[:max_items]


def collect_csd(max_items: int):
    snow_dir = DATA["csd"] / "Train" / "Snow"
    gt_dir = DATA["csd"] / "Train" / "Gt"
    gt_by_stem = {p.stem: p for p in list_images(gt_dir)}
    pairs = []
    for snow in list_images(snow_dir):
        gt = gt_by_stem.get(snow.stem) or gt_by_stem.get(snow.stem.split("_")[0])
        if gt:
            pairs.append((snow, gt))
        if len(pairs) >= max_items:
            break
    return pairs


def collect_val_pairs(max_items: int):
    return {
        "rain100h": list(zip(
            list_images(DATA["rain100h"] / "test" / "rain"),
            list_images(DATA["rain100h"] / "test" / "norain"),
        ))[:max_items],
        "rain100l": list(zip(
            list_images(DATA["rain100l"] / "test" / "rain"),
            list_images(DATA["rain100l"] / "test" / "norain"),
        ))[:max_items],
        "motion-blurry": read_csv_pairs(DATA["gopro"] / "gopro_test_pairs.csv")[:max_items],
        "noisy": read_csv_pairs(DATA["sidd"] / "sidd_test_pairs.csv")[:max_items],
        "snowy": collect_csd_test(max_items),
    }


def collect_csd_test(max_items: int):
    snow_dir = DATA["csd"] / "Test" / "Snow"
    gt_dir = DATA["csd"] / "Test" / "Gt"
    gt_by_stem = {p.stem: p for p in list_images(gt_dir)}
    pairs = []
    for snow in list_images(snow_dir):
        gt = gt_by_stem.get(snow.stem) or gt_by_stem.get(snow.stem.split("_")[0])
        if gt:
            pairs.append((snow, gt))
        if len(pairs) >= max_items:
            break
    return pairs


def materialize_split(root: Path, split: str, pairs_by_distortion: dict):
    for distortion, pairs in pairs_by_distortion.items():
        lq_dir = root / split / distortion / "LQ"
        gt_dir = root / split / distortion / "GT"
        lq_dir.mkdir(parents=True, exist_ok=True)
        gt_dir.mkdir(parents=True, exist_ok=True)
        for idx, (lq, gt) in enumerate(pairs):
            write_pair(Path(lq), Path(gt), lq_dir, gt_dir, distortion.replace("-", "_"), idx)
        print(f"[{split}] {distortion}: {len(pairs)} pairs")


def write_yaml(path: Path, name: str, data_root: Path, daclip_path: Path, batch_size: int, patch_size: int,
               niter: int, save_freq: int, print_freq: int, val_freq: int):
    text = f"""#### generated for VTN fair P2 DA-CLIP baseline
name: {name}
use_tb_logger: true
model: denoising
distortion: [{", ".join(DISTORTIONS)}]
gpu_ids: [0]

sde:
  max_sigma: 50
  T: 100
  schedule: cosine
  eps: 0.005

degradation:
  sigma: 25
  noise_type: G
  scale: 4

datasets:
  train:
    name: Train_Dataset
    mode: MD
    dataroot: {str(data_root / "train").replace("\\", "/")}
    use_shuffle: true
    n_workers: 0
    batch_size: {batch_size}
    patch_size: {patch_size}
    use_flip: true
    use_rot: true
    color: RGB
  val:
    name: Val_Dataset
    mode: MD
    dataroot: {str(data_root / "val").replace("\\", "/")}
    n_workers: 0
    batch_size: 1
    patch_size: {patch_size}
    use_flip: false
    use_rot: false
    color: RGB

network_G:
  which_model_G: ConditionalUNet
  setting:
    in_nc: 3
    out_nc: 3
    nf: 64
    ch_mult: [1, 2, 4, 8]
    context_dim: 512
    use_degra_context: true
    use_image_context: true

path:
  pretrain_model_G: ~
  strict_load: true
  resume_state: ~
  daclip: {str(daclip_path).replace("\\", "/")}

train:
  optimizer: AdamW
  lr_G: !!float 2e-4
  lr_scheme: TrueCosineAnnealingLR
  beta1: 0.9
  beta2: 0.99
  niter: {niter}
  warmup_iter: -1
  lr_steps: [200000, 400000, 600000]
  lr_gamma: 0.5
  eta_min: !!float 1e-6
  is_weighted: False
  loss_type: l1
  weight: 1.0
  manual_seed: 0
  val_freq: {val_freq}

logger:
  print_freq: {print_freq}
  save_checkpoint_freq: {save_freq}
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare DA-CLIP P2 all-in-one baseline data and option file.")
    parser.add_argument("--max-per-task", type=int, default=160)
    parser.add_argument("--val-per-task", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--niter", type=int, default=40000,
                        help="Default matches Restormer/PromptIR 100 epochs x 400 batches.")
    parser.add_argument("--save-freq", type=int, default=4000)
    parser.add_argument("--print-freq", type=int, default=200)
    parser.add_argument("--val-freq", type=int, default=100000000,
                        help="Large default avoids slow SDE validation during training.")
    parser.add_argument("--name", default="vtn_p2_daclip")
    parser.add_argument("--data-root", default=str(ROOT_DIR / "baselines_data" / "daclip_p2"))
    parser.add_argument("--daclip-weights", default=str(DA_CLIP_ROOT / "pretrained" / "daclip_ViT-B-32.pt"))
    parser.add_argument("--out", default=str(ROOT_DIR / "experiments" / "daclip_p2_train.yml"))
    return parser.parse_args()


def main():
    args = parse_args()
    if not DA_CLIP_CFG.exists():
        raise FileNotFoundError(f"DA-CLIP config directory not found: {DA_CLIP_CFG}")

    data_root = Path(args.data_root)
    train_pairs = {
        "rain100h": collect_rain(DATA["rain100h"], args.max_per_task),
        "rain100l": collect_rain(DATA["rain100l"], args.max_per_task),
        "motion-blurry": read_csv_pairs(DATA["gopro"] / "gopro_train_pairs.csv")[:args.max_per_task],
        "noisy": read_csv_pairs(DATA["sidd"] / "sidd_pairs.csv")[:args.max_per_task],
        "snowy": collect_csd(args.max_per_task),
    }
    val_pairs = collect_val_pairs(args.val_per_task)

    materialize_split(data_root, "train", train_pairs)
    materialize_split(data_root, "val", val_pairs)

    daclip_weights = Path(args.daclip_weights)
    write_yaml(
        Path(args.out),
        args.name,
        data_root,
        daclip_weights,
        args.batch_size,
        args.patch_size,
        args.niter,
        args.save_freq,
        args.print_freq,
        args.val_freq,
    )

    print(f"[Options] {args.out}")
    print(f"[Data] {data_root}")
    if not daclip_weights.exists():
        print(f"[Missing] DA-CLIP pretrained controller: {daclip_weights}")
        print("Download daclip_ViT-B-32.pt before training.")
    print("\nRun training:")
    print(f'python "{ROOT_DIR / "scripts" / "train_p2_daclip_baseline.py"}" --opt "{args.out}"')


if __name__ == "__main__":
    main()
