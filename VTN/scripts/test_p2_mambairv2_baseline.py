import argparse
import csv
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

CUR_DIR = Path(__file__).resolve().parent
ROOT_DIR = CUR_DIR.parent
WORKSPACE_DIR = ROOT_DIR.parent
MAMBAIR_DIR = WORKSPACE_DIR / "baselines" / "MambaIR"

for p in [ROOT_DIR, MAMBAIR_DIR]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import test_all_tasks as evaluator
from train_p2_mambairv2_baseline import build_mambairv2


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a fair P2 all-in-one MambaIRv2 baseline.")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--model-size", choices=["S", "L"], default="S")
    parser.add_argument("--result-name", default="p2_mambairv2_baseline")
    parser.add_argument("--table-out", default=None)
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument("--save-triplets", action="store_true")
    return parser.parse_args()


@torch.no_grad()
def eval_one_dataset_limited(model, name: str, max_items: int | None):
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


def main():
    args = parse_args()
    if not os.path.exists(args.ckpt):
        raise FileNotFoundError(args.ckpt)

    device = evaluator.DEVICE
    results_dir = evaluator.as_str(evaluator.result_dir(args.result_name))
    evaluator.RESULTS_DIR = results_dir
    evaluator.SAVE_TRIPLETS = args.save_triplets
    evaluator.USE_AMP = False

    method_name = f"MambaIRv2-{args.model_size}"
    print(f"[Device] {device}")
    print(f"[CKPT ] {args.ckpt}")
    print(f"[Save ] RESULTS_DIR={results_dir}")
    print(f"[Protocol] P2 all-in-one {method_name} baseline, official architecture retrained on the 5-task protocol.")

    model = build_mambairv2(args.model_size).to(device)
    state = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state, strict=True)
    model.eval()

    names = ["Rain100H", "Rain100L", "GoPro", "RESIDE", "CSD"]
    results = {}
    for name in names:
        psnr, ssim = eval_one_dataset_limited(model, name, args.max_items)
        results[name] = (psnr, ssim)

    rain_psnr = (results["Rain100H"][0] + results["Rain100L"][0]) / 2.0
    rain_ssim = (results["Rain100H"][1] + results["Rain100L"][1]) / 2.0
    avg_psnr = (rain_psnr + results["GoPro"][0] + results["RESIDE"][0] + results["CSD"][0]) / 4.0
    avg_ssim = (rain_ssim + results["GoPro"][1] + results["RESIDE"][1] + results["CSD"][1]) / 4.0

    print("\n==============================")
    print(f"P2 {method_name} Baseline Summary")
    print(f"Rain100H : PSNR {results['Rain100H'][0]:.2f} | SSIM {results['Rain100H'][1]:.4f}")
    print(f"Rain100L : PSNR {results['Rain100L'][0]:.2f} | SSIM {results['Rain100L'][1]:.4f}")
    print(f"Rain(avg): PSNR {rain_psnr:.2f} | SSIM {rain_ssim:.4f}")
    print(f"GoPro    : PSNR {results['GoPro'][0]:.2f} | SSIM {results['GoPro'][1]:.4f}")
    print(f"RESIDE   : PSNR {results['RESIDE'][0]:.2f} | SSIM {results['RESIDE'][1]:.4f}")
    print(f"CSD      : PSNR {results['CSD'][0]:.2f} | SSIM {results['CSD'][1]:.4f}")
    print("------------------------------")
    print("AVG(4 tasks: Rain+GoPro+RESIDE+CSD)")
    print(f"PSNR {avg_psnr:.2f} dB | SSIM {avg_ssim:.4f}")
    print("==============================\n")

    table_out = Path(args.table_out) if args.table_out else ROOT_DIR / "experiments" / "table1_mambairv2_row.csv"
    table_out.parent.mkdir(parents=True, exist_ok=True)
    training_data = "Rain100H, Rain100L, GoPro, RESIDE-6K, CSD"
    if args.max_items is not None:
        training_data += f"; interim eval N={args.max_items}"
    row = {
        "Method": method_name,
        "Status": "Retrained",
        "Direct Comparison?": "Yes",
        "Training Data": training_data,
        "Rain(avg)": f"{rain_psnr:.2f} / {rain_ssim:.4f}",
        "GoPro": f"{results['GoPro'][0]:.2f} / {results['GoPro'][1]:.4f}",
        "RESIDE-6K": f"{results['RESIDE'][0]:.2f} / {results['RESIDE'][1]:.4f}",
        "CSD": f"{results['CSD'][0]:.2f} / {results['CSD'][1]:.4f}",
        "Avg": f"{avg_psnr:.2f} / {avg_ssim:.4f}",
        "Checkpoint": args.ckpt,
        "Protocol": f"P2 all-in-one, official {method_name}, 5-task protocol, max_items={args.max_items}",
    }
    with open(table_out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)

    md_out = table_out.with_suffix(".md")
    with open(md_out, "w", encoding="utf-8") as f:
        f.write("| Method | Status | Direct Comparison? | Training Data | Rain(avg) | GoPro | RESIDE-6K | CSD | Avg |\n")
        f.write("|---|---|---|---|---:|---:|---:|---:|---:|\n")
        f.write(
            f"| {method_name} | Retrained | Yes | {training_data} | "
            f"{row['Rain(avg)']} | {row['GoPro']} | {row['RESIDE-6K']} | {row['CSD']} | {row['Avg']} |\n"
        )

    print(f"[Table 1 CSV] {table_out}")
    print(f"[Table 1 row] {md_out}")


if __name__ == "__main__":
    main()
