import csv
import os
import shutil
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
WORKSPACE_DIR = ROOT_DIR.parent
DIFFUIR_ROOT = WORKSPACE_DIR / "baselines" / "DiffUIR"

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import DATA

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def list_images(path: Path):
    return sorted([p for p in path.iterdir() if p.suffix.lower() in IMAGE_EXTS], key=lambda p: p.name.lower())


def link_or_copy(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def write_pair(src_lq: Path, src_gt: Path, lq_dir: Path, gt_dir: Path, name: str):
    suffix = src_lq.suffix.lower() if src_lq.suffix else ".png"
    link_or_copy(src_lq, lq_dir / f"{name}{suffix}")
    link_or_copy(src_gt, gt_dir / f"{name}{suffix}")


def read_csv_pairs(csv_path: Path):
    pairs = []
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if not row or len(row) < 2:
                continue
            if i == 0 and ("dist" in row[0].lower() or "blur" in row[0].lower()):
                continue
            lq = resolve_path(row[0])
            gt = resolve_path(row[1])
            if lq.exists() and gt.exists():
                pairs.append((lq, gt))
    return pairs


def resolve_path(raw):
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


def materialize_rain100(root: Path, split_name: str, input_dir: Path, target_dir: Path, max_items=None):
    lq = list_images(DATA["rain100h"] / "test" / "rain")
    gt = list_images(DATA["rain100h"] / "test" / "norain")
    pairs = list(zip(lq, gt))
    if max_items:
        pairs = pairs[:max_items]
    for idx, (a, b) in enumerate(pairs):
        write_pair(a, b, input_dir, target_dir, f"{split_name}_{idx:05d}")
    print(f"[rain] {len(pairs)} pairs -> {input_dir}")


def materialize_csv_pairs(csv_path: Path, input_dir: Path, target_dir: Path, prefix: str, max_items=None):
    pairs = read_csv_pairs(csv_path)
    if max_items:
        pairs = pairs[:max_items]
    for idx, (a, b) in enumerate(pairs):
        write_pair(a, b, input_dir, target_dir, f"{prefix}_{idx:05d}")
    print(f"[{prefix}] {len(pairs)} pairs -> {input_dir}")


def materialize_reside(input_dir: Path, target_dir: Path, max_items=None):
    hazy_dir = DATA["reside"] / "test" / "hazy"
    gt_dir = DATA["reside"] / "test" / "GT"
    gt_by_stem = {p.stem: p for p in list_images(gt_dir)}
    pairs = []
    for hazy in list_images(hazy_dir):
        gt = gt_by_stem.get(hazy.stem.split("_")[0]) or gt_by_stem.get(hazy.stem)
        if gt:
            pairs.append((hazy, gt))
        if max_items and len(pairs) >= max_items:
            break
    for idx, (a, b) in enumerate(pairs):
        write_pair(a, b, input_dir, target_dir, f"reside_{idx:05d}")
    print(f"[fog] {len(pairs)} pairs -> {input_dir}")


def materialize_csd(input_dir: Path, target_dir: Path, max_items=None):
    snow_dir = DATA["csd"] / "Test" / "Snow"
    gt_dir = DATA["csd"] / "Test" / "Gt"
    gt_by_stem = {p.stem: p for p in list_images(gt_dir)}
    pairs = []
    for snow in list_images(snow_dir):
        gt = gt_by_stem.get(snow.stem) or gt_by_stem.get(snow.stem.split("_")[0])
        if gt:
            pairs.append((snow, gt))
        if max_items and len(pairs) >= max_items:
            break
    for idx, (a, b) in enumerate(pairs):
        write_pair(a, b, input_dir, target_dir, f"csd_{idx:05d}")
    print(f"[snow] {len(pairs)} pairs -> {input_dir}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(ROOT_DIR / "baselines_data" / "diffuir_official"))
    parser.add_argument("--max-items", type=int, default=None)
    args = parser.parse_args()

    root = Path(args.root)

    materialize_rain100(
        root,
        "rain100h",
        root / "syn_rain" / "test" / "Test2800" / "input",
        root / "syn_rain" / "test" / "Test2800" / "target",
        args.max_items,
    )
    materialize_csv_pairs(
        DATA["gopro"] / "gopro_test_pairs.csv",
        root / "Deblur" / "test" / "GoPro" / "input",
        root / "Deblur" / "test" / "GoPro" / "target",
        "gopro",
        args.max_items,
    )
    materialize_reside(
        root / "RESIDE" / "SOTS" / "outdoor" / "hazy",
        root / "RESIDE" / "SOTS" / "outdoor" / "gt",
        args.max_items,
    )
    materialize_csd(
        root / "Snow100K" / "test" / "Snow100K-S" / "synthetic",
        root / "Snow100K" / "test" / "Snow100K-S" / "gt",
        args.max_items,
    )

    print(f"[DiffUIR layout] {root}")
    print("[Checkpoint expected]")
    print(DIFFUIR_ROOT / "ckpt_universal" / "diffuir" / "model-300.pt")


if __name__ == "__main__":
    main()
