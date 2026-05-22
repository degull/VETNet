import argparse
import csv
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

CUR_DIR = Path(__file__).resolve().parent
ROOT_DIR = CUR_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import test_all_tasks as evaluator
from config import CHECKPOINT_ROOT
from table5_ablation_common import VARIANTS, TABLE_ORDER, build_variant_model, latest_checkpoint


TASKS = ["Rain100H", "Rain100L", "GoPro", "RESIDE", "CSD"]


def parse_args():
    parser = argparse.ArgumentParser(description="Build Table 5 ablation study.")
    parser.add_argument("--variants", nargs="*", default=TABLE_ORDER)
    parser.add_argument("--max-items", type=int, default=None,
                        help="Evaluate only the first N samples per dataset for quick checks.")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--out-csv", default=str(ROOT_DIR / "experiments" / "table5_ablation.csv"))
    parser.add_argument("--out-md", default=str(ROOT_DIR / "experiments" / "table5_ablation.md"))
    return parser.parse_args()


@torch.no_grad()
def evaluate_dataset(model, name: str, max_items: int | None):
    ds, _ = evaluator.build_loader_for_dataset(name)
    if max_items is not None:
        ds = Subset(ds, list(range(min(max_items, len(ds)))))

    loader = DataLoader(
        ds,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=(evaluator.DEVICE.type == "cuda"),
    )

    psnr_sum, ssim_sum, n = 0.0, 0.0, 0
    pbar = tqdm(loader, desc=f"Testing {name}", leave=False)
    for inp, gt in pbar:
        inp = inp.to(evaluator.DEVICE, non_blocking=True)
        gt = gt.to(evaluator.DEVICE, non_blocking=True)
        out = evaluator.forward_with_optional_tile(model, inp)
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


def measure_efficiency(model, image_size: int):
    params = sum(p.numel() for p in model.parameters()) / 1e6
    flops = None
    try:
        from thop import profile
        x = torch.randn(1, 3, image_size, image_size, device=evaluator.DEVICE)
        macs, _ = profile(model, inputs=(x,), verbose=False)
        flops = macs / 1e9
    except Exception:
        pass
    return params, flops


def main():
    args = parse_args()
    evaluator.EVAL_SIZE = args.image_size
    evaluator.USE_AMP = args.amp
    evaluator.SAVE_TRIPLETS = False

    rows = []
    for variant in args.variants:
        cfg = VARIANTS[variant]
        row = {
            "Setting": cfg["label"],
            "Rain(avg)": "-",
            "GoPro": "-",
            "RESIDE-6K": "-",
            "CSD": "-",
            "Avg": "-",
            "Params": "-",
            "FLOPs": "-",
            "Checkpoint": "-",
            "Note": "",
        }

        ckpt = latest_checkpoint(CHECKPOINT_ROOT, variant)
        try:
            model = build_variant_model(variant).to(evaluator.DEVICE)
            if ckpt is None:
                params, flops = measure_efficiency(model, args.image_size)
                row["Params"] = f"{params:.2f}M"
                row["FLOPs"] = "-" if flops is None else f"{flops:.2f}G"
                row["Note"] = "checkpoint missing"
                rows.append(row)
                print(f"[Missing] {variant}: no checkpoint under {CHECKPOINT_ROOT / cfg['save_name']}")
                continue

            state = torch.load(ckpt, map_location=evaluator.DEVICE)
            model.load_state_dict(state, strict=True)
            model.eval()
            row["Checkpoint"] = str(ckpt)
            print(f"[Eval] {variant}: {ckpt}")

            params, flops = measure_efficiency(model, args.image_size)
            row["Params"] = f"{params:.2f}M"
            row["FLOPs"] = "-" if flops is None else f"{flops:.2f}G"

            scores = {}
            for task in TASKS:
                scores[task] = evaluate_dataset(model, task, args.max_items)

            rain_psnr = (scores["Rain100H"][0] + scores["Rain100L"][0]) / 2.0
            rain_ssim = (scores["Rain100H"][1] + scores["Rain100L"][1]) / 2.0
            avg_psnr = (rain_psnr + scores["GoPro"][0] + scores["RESIDE"][0] + scores["CSD"][0]) / 4.0
            avg_ssim = (rain_ssim + scores["GoPro"][1] + scores["RESIDE"][1] + scores["CSD"][1]) / 4.0

            row["Rain(avg)"] = metric_str(rain_psnr, rain_ssim)
            row["GoPro"] = metric_str(*scores["GoPro"])
            row["RESIDE-6K"] = metric_str(*scores["RESIDE"])
            row["CSD"] = metric_str(*scores["CSD"])
            row["Avg"] = metric_str(avg_psnr, avg_ssim)

        except Exception as exc:
            row["Note"] = f"{type(exc).__name__}: {exc}"
            print(f"[Fail] {variant}: {row['Note']}")
        finally:
            try:
                del model
            except Exception:
                pass
            if evaluator.DEVICE.type == "cuda":
                torch.cuda.empty_cache()

        rows.append(row)

    out_csv = Path(args.out_csv)
    out_md = Path(args.out_md)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = ["Setting", "Rain(avg)", "GoPro", "RESIDE-6K", "CSD", "Avg", "Params", "FLOPs", "Checkpoint", "Note"]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    with open(out_md, "w", encoding="utf-8") as f:
        f.write("| Setting | Rain(avg) | GoPro | RESIDE-6K | CSD | Avg | Params | FLOPs |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|\n")
        for row in rows:
            f.write(
                f"| {row['Setting']} | {row['Rain(avg)']} | {row['GoPro']} | {row['RESIDE-6K']} | "
                f"{row['CSD']} | {row['Avg']} | {row['Params']} | {row['FLOPs']} |\n"
            )

    print(f"[Table 5 CSV] {out_csv}")
    print(f"[Table 5 MD ] {out_md}")


if __name__ == "__main__":
    main()
