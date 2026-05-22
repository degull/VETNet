import argparse
import csv
import os
import sys
from pathlib import Path

import torch

CUR_DIR = Path(__file__).resolve().parent
ROOT_DIR = CUR_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from models.restormer_volterra import RestormerVolterra

# Reuse the exact P2 evaluator and datasets from the VTN all-in-one script.
import test_all_tasks as evaluator


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a P2 all-in-one Restormer baseline.")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--result-name", default="p2_restormer_baseline")
    parser.add_argument("--table-out", default=None,
                        help="Optional CSV path for the Table 1 Restormer row.")
    return parser.parse_args()


def main():
    args = parse_args()
    if not os.path.exists(args.ckpt):
        raise FileNotFoundError(args.ckpt)

    device = evaluator.DEVICE
    results_dir = evaluator.as_str(evaluator.result_dir(args.result_name))
    evaluator.RESULTS_DIR = results_dir

    print(f"[Device] {device}")
    print(f"[CKPT ] {args.ckpt}")
    print(f"[Save ] RESULTS_DIR={results_dir}")
    print("[Protocol] P2 all-in-one Restormer baseline, Volterra disabled in MDTA and FFN.")

    model = RestormerVolterra(use_volterra_mdta=False, use_volterra_gdfn=False).to(device)
    state = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state, strict=True)
    model.eval()

    names = ["Rain100H", "Rain100L", "GoPro", "RESIDE", "CSD"]
    results = {}
    for name in names:
        psnr, ssim = evaluator.eval_one_dataset(model, name)
        results[name] = (psnr, ssim)

    rain_psnr = (results["Rain100H"][0] + results["Rain100L"][0]) / 2.0
    rain_ssim = (results["Rain100H"][1] + results["Rain100L"][1]) / 2.0
    avg_psnr = (rain_psnr + results["GoPro"][0] + results["RESIDE"][0] + results["CSD"][0]) / 4.0
    avg_ssim = (rain_ssim + results["GoPro"][1] + results["RESIDE"][1] + results["CSD"][1]) / 4.0

    print("\n==============================")
    print("P2 Restormer Baseline Summary")
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

    table_out = Path(args.table_out) if args.table_out else ROOT_DIR / "experiments" / "table1_restormer_row.csv"
    table_out.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "Method": "Restormer",
        "Status": "Retrained",
        "Direct Comparison?": "Yes",
        "Training Data": "Same as VTN",
        "Rain(avg)": f"{rain_psnr:.2f} / {rain_ssim:.4f}",
        "GoPro": f"{results['GoPro'][0]:.2f} / {results['GoPro'][1]:.4f}",
        "RESIDE-6K": f"{results['RESIDE'][0]:.2f} / {results['RESIDE'][1]:.4f}",
        "CSD": f"{results['CSD'][0]:.2f} / {results['CSD'][1]:.4f}",
        "Avg": f"{avg_psnr:.2f} / {avg_ssim:.4f}",
        "Checkpoint": args.ckpt,
        "Protocol": "P2 all-in-one, Volterra disabled, same balanced160 protocol as VTN",
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
            f"| Restormer | Retrained | Yes | Same as VTN | "
            f"{row['Rain(avg)']} | {row['GoPro']} | {row['RESIDE-6K']} | {row['CSD']} | {row['Avg']} |\n"
        )

    print(f"[Table 1 CSV] {table_out}")
    print(f"[Table 1 row] {md_out}")


if __name__ == "__main__":
    main()
