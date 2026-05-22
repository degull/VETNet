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
from models.restormer_volterra import RestormerVolterra


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

TASKS = ["Rain100H", "Rain100L", "GoPro", "RESIDE", "CSD"]

VTN_TASK_CKPTS = {
    "Rain100H": ROOT_DIR / "checkpoints" / "restormer_volterra_rain100h" / "epoch_100.pth",
    "Rain100L": ROOT_DIR / "checkpoints" / "restormer_volterra_rain100l" / "epoch_91_ssim0.9538_psnr33.98.pth",
    "GoPro": ROOT_DIR / "checkpoints" / "restormer_volterra_gopro" / "epoch_013_ssim0.9690_psnr34.80.pth",
    "RESIDE": ROOT_DIR / "checkpoints" / "restormer_volterra_reside" / "epoch_015_ssim0.9520_psnr27.97.pth",
    "CSD": ROOT_DIR / "checkpoints" / "restormer_volterra_csd" / "epoch_5_ssim0.9531_psnr33.03.pth",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build Table 2 with reported baseline values and VTN task-specific evaluation."
    )
    parser.add_argument(
        "--reported-csv",
        default=str(ROOT_DIR / "experiments" / "table2_reported_baselines.csv"),
        help="CSV containing reported baseline values. Created as a blank template if missing.",
    )
    parser.add_argument(
        "--out-csv",
        default=str(ROOT_DIR / "experiments" / "table2_single_task.csv"),
    )
    parser.add_argument(
        "--out-md",
        default=str(ROOT_DIR / "experiments" / "table2_single_task.md"),
    )
    parser.add_argument(
        "--max-items",
        type=int,
        default=None,
        help="Evaluate only the first N samples per task for quick checks.",
    )
    parser.add_argument(
        "--skip-vtn-eval",
        action="store_true",
        help="Only create/merge the reported baseline table template without evaluating VTN.",
    )
    parser.add_argument(
        "--save-triplets",
        action="store_true",
        help="Save visual triplets through the shared evaluator.",
    )
    return parser.parse_args()


def ensure_reported_template(path: Path):
    if path.exists():
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["Method", "Source"] + TASKS
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for method in METHODS:
            if method == "VTN":
                continue
            row = {"Method": method, "Source": "reported"}
            for task in TASKS:
                row[task] = "-"
            writer.writerow(row)


def read_reported_rows(path: Path):
    ensure_reported_template(path)
    rows = {}
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            method = row.get("Method", "").strip()
            if not method:
                continue
            rows[method] = {
                "Method": method,
                "Source": row.get("Source", "reported").strip() or "reported",
                **{task: row.get(task, "-").strip() or "-" for task in TASKS},
            }
    return rows


@torch.no_grad()
def evaluate_task(model: torch.nn.Module, task: str, max_items: int | None):
    ds, _ = evaluator.build_loader_for_dataset(task)
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
    pbar = tqdm(loader, desc=f"VTN single-task {task}", leave=False)
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


def evaluate_vtn(max_items: int | None, save_triplets: bool):
    evaluator.USE_AMP = True
    evaluator.SAVE_TRIPLETS = save_triplets
    evaluator.RESULTS_DIR = evaluator.as_str(evaluator.result_dir("table2_vtn_single_task"))

    results = {}
    for task in TASKS:
        ckpt = VTN_TASK_CKPTS[task]
        if not ckpt.exists():
            raise FileNotFoundError(f"Missing VTN task-specific checkpoint for {task}: {ckpt}")

        print(f"[VTN] {task} checkpoint: {ckpt}")
        model = RestormerVolterra().to(evaluator.DEVICE)
        state = torch.load(ckpt, map_location=evaluator.DEVICE)
        model.load_state_dict(state, strict=True)
        model.eval()

        psnr, ssim = evaluate_task(model, task, max_items)
        results[task] = f"{psnr:.2f} / {ssim:.4f}"
        del model
        if evaluator.DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    return {
        "Method": "VTN",
        "Source": "ours, task-specific",
        **results,
    }


def write_outputs(rows, out_csv: Path, out_md: Path):
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = ["Method", "Source"] + TASKS
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    with open(out_md, "w", encoding="utf-8") as f:
        f.write("| Method | Source | Rain100H | Rain100L | GoPro | RESIDE-6K | CSD |\n")
        f.write("|---|---|---:|---:|---:|---:|---:|\n")
        for row in rows:
            f.write(
                f"| {row['Method']} | {row['Source']} | {row['Rain100H']} | {row['Rain100L']} | "
                f"{row['GoPro']} | {row['RESIDE']} | {row['CSD']} |\n"
            )


def main():
    args = parse_args()
    reported_csv = Path(args.reported_csv)
    reported_rows = read_reported_rows(reported_csv)
    previous_rows = {}
    out_csv = Path(args.out_csv)
    if out_csv.exists():
        with open(out_csv, "r", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                previous_rows[row.get("Method", "")] = row

    rows = []
    for method in METHODS:
        if method == "VTN":
            if args.skip_vtn_eval:
                previous = previous_rows.get("VTN")
                if previous and all(previous.get(task) not in (None, "", "to evaluate") for task in TASKS):
                    row = {
                        "Method": "VTN",
                        "Source": previous.get("Source", "ours, task-specific"),
                        **{task: previous.get(task, "to evaluate") for task in TASKS},
                    }
                else:
                    row = {"Method": "VTN", "Source": "ours, task-specific"}
                    row.update({task: "to evaluate" for task in TASKS})
            else:
                row = evaluate_vtn(args.max_items, args.save_triplets)
        else:
            row = reported_rows.get(method)
            if row is None:
                row = {"Method": method, "Source": "reported", **{task: "-" for task in TASKS}}
        rows.append(row)

    write_outputs(rows, out_csv, Path(args.out_md))
    print(f"[Reported template] {reported_csv}")
    print(f"[Table 2 CSV] {args.out_csv}")
    print(f"[Table 2 MD ] {args.out_md}")


if __name__ == "__main__":
    main()
