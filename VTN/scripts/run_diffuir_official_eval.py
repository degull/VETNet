import argparse
import csv
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parent.parent
WORKSPACE_DIR = ROOT_DIR.parent
DIFFUIR_ROOT = WORKSPACE_DIR / "baselines" / "DiffUIR"
DIFFUIR_DEPS = ROOT_DIR / ".deps_diffuir"

if DIFFUIR_DEPS.exists() and str(DIFFUIR_DEPS) not in sys.path:
    sys.path.insert(0, str(DIFFUIR_DEPS))
if str(DIFFUIR_ROOT) not in sys.path:
    sys.path.insert(0, str(DIFFUIR_ROOT))

from data.universal_dataset import AlignedDataset_all
import src.model as diffuir_model
from src.model import ResidualDiffusion, Trainer, UnetRes, metric_module, set_seed, tensor2img


def quiet_tqdm(iterable=None, *args, **kwargs):
    kwargs["disable"] = True
    return tqdm(iterable, *args, **kwargs)


diffuir_model.tqdm = quiet_tqdm


class Opt:
    def __init__(self, dataroot, phase="test", max_dataset_size=float("inf")):
        self.dataroot = dataroot
        self.phase = phase
        self.max_dataset_size = max_dataset_size
        self.load_size = 256
        self.crop_size = 256
        self.direction = "AtoB"
        self.preprocess = "none"
        self.no_flip = True
        self.bsize = 1


def build_trainer(dataset, ckpt_dir, sampling_timesteps=3):
    condition = True
    image_size = 256
    num_unet = 1
    objective = "pred_res"
    test_res_or_noise = "res"
    sum_scale = 0.01
    ddim_sampling_eta = 0.0
    delta_end = 1.8e-3

    model = UnetRes(
        dim=64,
        dim_mults=(1, 2, 2, 4),
        num_unet=num_unet,
        condition=condition,
        objective=objective,
        test_res_or_noise=test_res_or_noise,
    )
    diffusion = ResidualDiffusion(
        model,
        image_size=image_size,
        timesteps=1000,
        delta_end=delta_end,
        sampling_timesteps=sampling_timesteps,
        ddim_sampling_eta=ddim_sampling_eta,
        objective=objective,
        loss_type="l1",
        condition=condition,
        sum_scale=sum_scale,
        test_res_or_noise=test_res_or_noise,
    )

    return Trainer(
        diffusion,
        dataset,
        dataset.opt,
        train_batch_size=1,
        num_samples=1,
        train_lr=2e-4,
        train_num_steps=100000,
        gradient_accumulate_every=2,
        ema_decay=0.995,
        amp=False,
        convert_image_to="RGB",
        results_folder=str(ckpt_dir),
        condition=condition,
        save_and_sample_every=1000,
        num_unet=num_unet,
    )


def eval_task(task, dataroot, ckpt_dir, milestone, max_items=None):
    opt = Opt(dataroot=dataroot, max_dataset_size=max_items if max_items is not None else float("inf"))
    dataset = AlignedDataset_all(
        opt,
        image_size=256,
        augment_flip=False,
        equalizeHist=True,
        crop_patch=False,
        generation=False,
        task=task,
    )
    trainer = build_trainer(dataset, ckpt_dir)
    trainer.load(milestone)
    trainer.ema.ema_model.init()
    trainer.ema.to(trainer.device)
    trainer.ema.ema_model.eval()

    loader = DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=0)
    psnr_sum = 0.0
    ssim_sum = 0.0
    count = 0

    for items in tqdm(loader, desc=f"DiffUIR {task}", ncols=100):
        file_path = items["A_paths"][0]
        x_input_sample = items["adap"].to(trainer.device)
        gt = items["gt"].to(trainer.device)

        with torch.no_grad():
            samples = list(
                trainer.ema.ema_model.sample(
                    x_input_sample,
                    batch_size=trainer.num_samples,
                    last=True,
                    task=file_path,
                )
            )
        sr_img = tensor2img(samples[-1], rgb2bgr=True)
        gt_img = tensor2img(gt, rgb2bgr=True)
        psnr_sum += metric_module.calculate_psnr(sr_img, gt_img, crop_border=0, test_y_channel=True)
        ssim_sum += metric_module.calculate_ssim(sr_img, gt_img, crop_border=0, test_y_channel=True)
        count += 1

    return psnr_sum / count, ssim_sum / count, count


def parse_args():
    parser = argparse.ArgumentParser(description="Run official DiffUIR-B model-300 evaluation.")
    parser.add_argument("--dataroot", default=str(ROOT_DIR / "baselines_data" / "diffuir_official"))
    parser.add_argument("--ckpt-dir", default=str(DIFFUIR_ROOT / "ckpt_universal" / "diffuir"))
    parser.add_argument("--milestone", type=int, default=300)
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument("--method", default="DiffUIR")
    parser.add_argument("--status", default="Official / Evaluated")
    parser.add_argument("--direct-comparison", default="No")
    parser.add_argument("--training-data", default=None)
    parser.add_argument("--row-name", default="diffuir")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(10)
    ckpt_dir = Path(args.ckpt_dir)
    ckpt = ckpt_dir / f"model-{args.milestone}.pt"
    if not ckpt.exists():
        raise FileNotFoundError(
            f"DiffUIR official checkpoint is missing: {ckpt}\n"
            f"Place the target checkpoint as model-{args.milestone}.pt or pass --ckpt-dir/--milestone."
        )

    task_map = {
        "GoPro": "blur",
        "Rain(avg)": "rain",
        "RESIDE-6K": "fog",
        "CSD": "snow",
    }
    metrics = {}
    for label, task in task_map.items():
        psnr, ssim, n = eval_task(task, args.dataroot, ckpt_dir, args.milestone, args.max_items)
        metrics[label] = (psnr, ssim, n)
        print(f"[{label}] task={task} N={n} PSNR={psnr:.2f} SSIM={ssim:.4f}")

    avg_psnr = (metrics["Rain(avg)"][0] + metrics["GoPro"][0] + metrics["RESIDE-6K"][0] + metrics["CSD"][0]) / 4
    avg_ssim = (metrics["Rain(avg)"][1] + metrics["GoPro"][1] + metrics["RESIDE-6K"][1] + metrics["CSD"][1]) / 4
    if args.training_data:
        training_data = args.training_data
    else:
        suffix = f"; model-{args.milestone} (N={args.max_items})" if args.max_items is not None else f"; model-{args.milestone}"
        training_data = "Official protocol" + suffix

    row = {
        "Method": args.method,
        "Status": args.status,
        "Direct Comparison?": args.direct_comparison,
        "Training Data": training_data,
        "Rain(avg)": f"{metrics['Rain(avg)'][0]:.2f} / {metrics['Rain(avg)'][1]:.4f}",
        "GoPro": f"{metrics['GoPro'][0]:.2f} / {metrics['GoPro'][1]:.4f}",
        "RESIDE-6K": f"{metrics['RESIDE-6K'][0]:.2f} / {metrics['RESIDE-6K'][1]:.4f}",
        "CSD": f"{metrics['CSD'][0]:.2f} / {metrics['CSD'][1]:.4f}",
        "Avg": f"{avg_psnr:.2f} / {avg_ssim:.4f}",
        "Checkpoint": str(ckpt),
        "Protocol": f"DiffUIR-B/Base model-{args.milestone}; sampling timestep=3; max_items={args.max_items}",
    }

    out_csv = ROOT_DIR / "experiments" / f"table1_{args.row_name}_row.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)

    out_md = out_csv.with_suffix(".md")
    with out_md.open("w", encoding="utf-8") as f:
        f.write("| Method | Status | Direct Comparison? | Training Data | Rain(avg) | GoPro | RESIDE-6K | CSD | Avg |\n")
        f.write("|---|---|---|---|---:|---:|---:|---:|---:|\n")
        f.write(
            f"| {row['Method']} | {row['Status']} | {row['Direct Comparison?']} | {row['Training Data']} | "
            f"{row['Rain(avg)']} | {row['GoPro']} | {row['RESIDE-6K']} | {row['CSD']} | {row['Avg']} |\n"
        )
    print(f"[Table 1 CSV] {out_csv}")
    print(f"[Table 1 row] {out_md}")


if __name__ == "__main__":
    main()
