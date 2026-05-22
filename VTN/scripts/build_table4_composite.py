import argparse
import csv
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from torchvision.transforms.functional import gaussian_blur
from tqdm import tqdm

CUR_DIR = Path(__file__).resolve().parent
ROOT_DIR = CUR_DIR.parent
WORKSPACE_DIR = ROOT_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import test_all_tasks as evaluator
from config import DATA, as_str
from datasets.rain100h_dataset import Rain100HDataset
from datasets.rain100l_dataset import Rain100LDataset
from datasets.csd_dataset import CSDDataset
from datasets.kadis700k_dataset import KADIS700KCSVDataset
from models.restormer_volterra import RestormerVolterra
from scripts.build_table3_efficiency import (
    build_promptir,
    build_adair,
    build_moceir,
    build_hint,
    build_diffuir,
)


METHODS = [
    "Restormer",
    "PromptIR",
    "DiffUIR",
    "DA-CLIP",
    "AdaIR",
    "MoCE-IR",
    "MambaIRv2",
    "HINT",
    "VTN",
]

CKPTS = {
    "Restormer": ROOT_DIR / "checkpoints" / "restormer_p2_all_in_one" / "epoch_100_loss0.02006.pth",
    "PromptIR": ROOT_DIR / "checkpoints" / "promptir_p2_all_in_one" / "epoch_100_loss0.01982.pth",
    "DiffUIR": ROOT_DIR / "checkpoints" / "diffuir_p2_all_in_one" / "model-100.pt",
    "AdaIR": ROOT_DIR / "checkpoints" / "adair_p2_all_in_one" / "epoch_029_loss0.02706.pth",
    "MoCE-IR": ROOT_DIR / "checkpoints" / "moceir_s_p2_reside_all_in_one" / "epoch_100_loss0.03531.pth",
    "HINT": ROOT_DIR / "checkpoints" / "hint_s_p2_reside_all_in_one" / "epoch_100_loss0.03131.pth",
    "VTN": ROOT_DIR / "checkpoints" / "#01_all_tasks_balanced_160" / "epoch_99_ssim0.9183_psnr32.73.pth",
}


class CompositeDataset(Dataset):
    def __init__(self, base_dataset, recipe: str):
        self.base = base_dataset
        self.recipe = recipe

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        inp, gt = self.base[idx]
        if self.recipe == "rain_haze":
            inp = apply_haze(inp, strength=0.28, airlight=0.92)
        elif self.recipe == "rain_blur":
            inp = gaussian_blur(inp, kernel_size=[13, 13], sigma=[2.0, 2.0])
        elif self.recipe == "haze_snow":
            inp = apply_haze(inp, strength=0.24, airlight=0.92)
        else:
            raise ValueError(f"Unknown composite recipe: {self.recipe}")
        return inp.clamp(0, 1), gt


def apply_haze(x: torch.Tensor, strength: float, airlight: float):
    return x * (1.0 - strength) + airlight * strength


def build_composite_dataset(recipe: str, image_size: int, rain_source: str):
    tfm = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])
    if recipe in ("rain_haze", "rain_blur"):
        if rain_source == "Rain100L":
            base = Rain100LDataset(as_str(DATA["rain100l"] / "test"), transform=tfm)
        else:
            base = Rain100HDataset(as_str(DATA["rain100h"] / "test"), transform=tfm)
    elif recipe == "haze_snow":
        base = CSDDataset(root_dir=as_str(DATA["csd"]), split="test", transform=tfm)
    else:
        raise ValueError(recipe)
    return CompositeDataset(base, recipe)


def build_model(method: str, device):
    if method == "Restormer":
        model = RestormerVolterra(use_volterra_mdta=False, use_volterra_gdfn=False)
    elif method == "PromptIR":
        model = build_promptir()
    elif method == "DiffUIR":
        model = build_diffuir()
    elif method == "AdaIR":
        model = build_adair()
    elif method == "MoCE-IR":
        model = build_moceir()
    elif method == "HINT":
        model = build_hint()
    elif method == "VTN":
        model = RestormerVolterra()
    else:
        raise RuntimeError(f"No local runnable builder for {method}")

    model = model.to(device)
    ckpt = CKPTS.get(method)
    if ckpt is None or not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found for {method}: {ckpt}")

    state = torch.load(ckpt, map_location=device)
    if method == "DiffUIR":
        if isinstance(state, dict) and "model" in state:
            model.diffusion.load_state_dict(state["model"], strict=True)
        else:
            model.diffusion.load_state_dict(state, strict=True)
    else:
        model.load_state_dict(state, strict=True)
    model.eval()
    return model, ckpt


@torch.no_grad()
def forward_model(method: str, model, inp):
    if method == "HINT":
        model.qv_cache = None
    out = model(inp)
    if isinstance(out, (list, tuple)):
        out = out[-1]
    return out.clamp(0, 1)


@torch.no_grad()
def evaluate_loader(method: str, model, loader, device, amp: bool):
    psnr_sum, ssim_sum, n = 0.0, 0.0, 0
    pbar = tqdm(loader, desc=f"{method}", leave=False)
    for inp, gt in pbar:
        inp = inp.to(device, non_blocking=True)
        gt = gt.to(device, non_blocking=True)
        with torch.amp.autocast(device_type="cuda", enabled=(amp and device.type == "cuda")):
            out = forward_model(method, model, inp)

        out_np = evaluator.tensor_to_hwc01(out)
        gt_np = evaluator.tensor_to_hwc01(gt)
        psnr = evaluator.compute_psnr(gt_np, out_np, data_range=1.0)
        ssim = evaluator.compute_ssim(gt_np, out_np, channel_axis=2, data_range=1.0, win_size=7)
        psnr_sum += psnr
        ssim_sum += ssim
        n += 1
        pbar.set_postfix(psnr=f"{psnr:.2f}", ssim=f"{ssim:.4f}")
    return psnr_sum / max(n, 1), ssim_sum / max(n, 1)


def metric_str(psnr, ssim):
    return f"{psnr:.2f} / {ssim:.4f}"


def parse_args():
    parser = argparse.ArgumentParser(description="Build Table 4 composite degradation / robustness results.")
    parser.add_argument("--methods", nargs="*", default=METHODS)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--max-items", type=int, default=100,
                        help="Number of samples per composite set. Use -1 for full.")
    parser.add_argument("--rain-source", choices=["Rain100H", "Rain100L"], default="Rain100H")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--kadis-root", default=str(DATA.get("kadis", ROOT_DIR / "data" / "KADIS-700K")))
    parser.add_argument("--kadis-csv", default=None,
                        help="Optional KADIS-700K paired CSV. If omitted, KADIS table is left blank.")
    parser.add_argument("--out-left-csv", default=str(ROOT_DIR / "experiments" / "table4_composite_left.csv"))
    parser.add_argument("--out-right-csv", default=str(ROOT_DIR / "experiments" / "table4_kadis_right.csv"))
    parser.add_argument("--out-md", default=str(ROOT_DIR / "experiments" / "table4_composite_robustness.md"))
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    max_items = None if args.max_items is not None and args.max_items < 0 else args.max_items

    recipes = [
        ("Rain+Haze", "rain_haze"),
        ("Rain+Blur", "rain_blur"),
        ("Haze+Snow", "haze_snow"),
    ]
    recipe_loaders = {}
    for label, recipe in recipes:
        ds = build_composite_dataset(recipe, args.image_size, args.rain_source)
        if max_items is not None:
            ds = Subset(ds, list(range(min(max_items, len(ds)))))
        recipe_loaders[label] = DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=(device.type == "cuda"),
        )

    kadis_loader = None
    if args.kadis_csv:
        tfm = transforms.Compose([
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor(),
        ])
        kadis_ds = KADIS700KCSVDataset(
            root_dir=args.kadis_root,
            csv_path=args.kadis_csv,
            transform=tfm,
            strict=True,
        )
        if max_items is not None:
            kadis_ds = Subset(kadis_ds, list(range(min(max_items, len(kadis_ds)))))
        kadis_loader = DataLoader(
            kadis_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=(device.type == "cuda"),
        )

    left_rows = []
    right_rows = []
    for method in args.methods:
        left_row = {
            "Method": method,
            "Rain+Haze": "-",
            "Rain+Blur": "-",
            "Haze+Snow": "-",
            "Composite Avg": "-",
            "Checkpoint": str(CKPTS.get(method, "-")),
            "Note": "",
        }
        right_row = {
            "Method": method,
            "KADIS-700K Composite": "Mixed degradation" if kadis_loader is not None else "-",
            "PSNR": "-",
            "SSIM": "-",
            "Checkpoint": str(CKPTS.get(method, "-")),
            "Note": "",
        }
        try:
            model, ckpt = build_model(method, device)
            scores = []
            print(f"[Model] {method} | {ckpt}")
            for label, loader in recipe_loaders.items():
                psnr, ssim = evaluate_loader(method, model, loader, device, args.amp)
                left_row[label] = metric_str(psnr, ssim)
                scores.append((psnr, ssim))
                print(f"  {label}: {left_row[label]}")

            avg_psnr = sum(s[0] for s in scores) / max(len(scores), 1)
            avg_ssim = sum(s[1] for s in scores) / max(len(scores), 1)
            left_row["Composite Avg"] = metric_str(avg_psnr, avg_ssim)

            if kadis_loader is not None:
                psnr, ssim = evaluate_loader(method, model, kadis_loader, device, args.amp)
                right_row["PSNR"] = f"{psnr:.2f}"
                right_row["SSIM"] = f"{ssim:.4f}"

        except Exception as exc:
            note = f"{type(exc).__name__}: {exc}"
            left_row["Note"] = note
            right_row["Note"] = note
            print(f"[Skip/Fail] {method}: {note}")
        finally:
            try:
                del model
            except Exception:
                pass
            if device.type == "cuda":
                torch.cuda.empty_cache()

        left_rows.append(left_row)
        right_rows.append(right_row)

    out_left = Path(args.out_left_csv)
    out_right = Path(args.out_right_csv)
    out_md = Path(args.out_md)
    out_left.parent.mkdir(parents=True, exist_ok=True)

    with open(out_left, "w", newline="", encoding="utf-8") as f:
        fields = ["Method", "Rain+Haze", "Rain+Blur", "Haze+Snow", "Composite Avg", "Checkpoint", "Note"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(left_rows)

    with open(out_right, "w", newline="", encoding="utf-8") as f:
        fields = ["Method", "KADIS-700K Composite", "PSNR", "SSIM", "Checkpoint", "Note"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(right_rows)

    with open(out_md, "w", encoding="utf-8") as f:
        f.write("## Table 4(a). Composite Degradation\n\n")
        f.write("| Method | Rain+Haze | Rain+Blur | Haze+Snow | Composite Avg |\n")
        f.write("|---|---:|---:|---:|---:|\n")
        for row in left_rows:
            f.write(
                f"| {row['Method']} | {row['Rain+Haze']} | {row['Rain+Blur']} | "
                f"{row['Haze+Snow']} | {row['Composite Avg']} |\n"
            )
        f.write("\n## Table 4(b). KADIS-700K Composite\n\n")
        f.write("| Method | KADIS-700K Composite | PSNR | SSIM |\n")
        f.write("|---|---|---:|---:|\n")
        for row in right_rows:
            f.write(
                f"| {row['Method']} | {row['KADIS-700K Composite']} | {row['PSNR']} | {row['SSIM']} |\n"
            )

    print(f"[Table 4 left CSV ] {out_left}")
    print(f"[Table 4 right CSV] {out_right}")
    print(f"[Table 4 MD       ] {out_md}")


if __name__ == "__main__":
    main()
