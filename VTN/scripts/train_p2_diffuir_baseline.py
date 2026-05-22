import argparse
import sys
import time
from pathlib import Path

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

CUR_DIR = Path(__file__).resolve().parent
ROOT_DIR = CUR_DIR.parent
WORKSPACE_DIR = ROOT_DIR.parent
DIFFUIR_ROOT = WORKSPACE_DIR / "baselines" / "DiffUIR"
DIFFUIR_DEPS = ROOT_DIR / ".deps_diffuir"

for p in [DIFFUIR_DEPS, ROOT_DIR, DIFFUIR_ROOT]:
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))

from config import CHECKPOINT_ROOT, result_dir
from ema_pytorch import EMA
from src.model import ResidualDiffusion, UnetRes, set_seed
from train_p2_restormer_baseline import build_train_dataset, scheduled_size


def build_diffuir_b(sampling_timesteps=3):
    condition = True
    objective = "pred_res"
    test_res_or_noise = "res"
    sum_scale = 0.01
    delta_end = 1.8e-3

    model = UnetRes(
        dim=64,
        dim_mults=(1, 2, 2, 4),
        num_unet=1,
        condition=condition,
        objective=objective,
        test_res_or_noise=test_res_or_noise,
    )
    return ResidualDiffusion(
        model,
        image_size=256,
        timesteps=1000,
        delta_end=delta_end,
        sampling_timesteps=sampling_timesteps,
        ddim_sampling_eta=0.0,
        objective=objective,
        loss_type="l1",
        condition=condition,
        sum_scale=sum_scale,
        test_res_or_noise=test_res_or_noise,
    )


def load_checkpoint(path, diffusion, optimizer, ema, device):
    data = torch.load(path, map_location=device)
    if isinstance(data, dict) and "model" in data:
        diffusion.load_state_dict(data["model"], strict=True)
        if optimizer is not None and "opt0" in data:
            optimizer.load_state_dict(data["opt0"])
        if ema is not None and "ema" in data:
            ema.load_state_dict(data["ema"])
        return int(data.get("step", 0))
    diffusion.load_state_dict(data, strict=True)
    return 0


def parse_args():
    parser = argparse.ArgumentParser(description="Train a fair P2 all-in-one DiffUIR-B baseline.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--start-epoch", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=8e-5)
    parser.add_argument("--max-per-task", type=int, default=160,
                        help="Samples per task per epoch. Default 160 matches VTN unified_balanced160.")
    parser.add_argument("--save-name", default="diffuir_p2_all_in_one")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--sampling-timesteps", type=int, default=3)
    parser.add_argument("--ema-update-every", type=int, default=10)
    parser.add_argument("--ema-decay", type=float, default=0.995)
    parser.add_argument("--seed", type=int, default=10)
    return parser.parse_args()


def main():
    args = parse_args()
    if not DIFFUIR_ROOT.exists():
        raise FileNotFoundError(f"DiffUIR repo was not found: {DIFFUIR_ROOT}")

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = CHECKPOINT_ROOT / args.save_name
    save_dir.mkdir(parents=True, exist_ok=True)
    log_dir = result_dir(f"train_{args.save_name}")

    diffusion = build_diffuir_b(args.sampling_timesteps).to(device)
    optimizer = Adam(diffusion.parameters(), lr=args.lr, betas=(0.9, 0.99))
    ema = EMA(diffusion, beta=args.ema_decay, update_every=args.ema_update_every).to(device)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))
    global_step = 0

    if args.resume:
        global_step = load_checkpoint(args.resume, diffusion, optimizer, ema, device)
        print(f"[Resume] {args.resume} | step={global_step}")

    print(f"[Device] {device}")
    print(f"[Save] {save_dir}")
    print("[Protocol] P2 all-in-one DiffUIR-B baseline.")
    print("[Protocol] Official DiffUIR-B architecture: dim=64, dim_mults=(1,2,2,4), objective=pred_res.")
    print("[Protocol] Retrained on the revised Table 1 mixture: Rain100H, Rain100L, GoPro, RESIDE-6K, CSD.")
    print("[Protocol] Samples/task/epoch=160 by default; resize schedule: epoch 1-29=128, 30-59=192, 60+=256.")

    with open(log_dir / "protocol.txt", "w", encoding="utf-8") as f:
        f.write("P2 all-in-one DiffUIR-B baseline for Table 1\n")
        f.write("Direct comparison target: VTN --ckpt-key unified_balanced160\n")
        f.write("Architecture: official DiffUIR-B, 64-1224\n")
        f.write("Initialization: from scratch unless --resume is provided\n")
        f.write("Tasks: Rain100H, Rain100L, GoPro, RESIDE-6K, CSD\n")
        f.write(f"Samples per task per epoch: {args.max_per_task}\n")
        f.write("Resize schedule: epoch 1-29=128, 30-59=192, 60+=256\n")
        f.write(f"Batch size: {args.batch_size}\n")
        f.write(f"LR: {args.lr}\n")
        f.write(f"Sampling timesteps for saved model config: {args.sampling_timesteps}\n")

    for epoch in range(args.start_epoch, args.epochs + 1):
        image_size = scheduled_size(epoch)
        dataset = build_train_dataset(image_size, args.max_per_task)
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=(device.type == "cuda"),
        )

        diffusion.train()
        t0 = time.time()
        total_loss = 0.0
        print(f"[Epoch {epoch}] image_size={image_size} samples={len(dataset)}")

        for inp, gt in tqdm(loader, desc=f"Epoch {epoch}/{args.epochs}"):
            inp = inp.to(device, non_blocking=True)
            gt = gt.to(device, non_blocking=True)
            task = ["p2_all_in_one"] * inp.shape[0]

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type="cuda", enabled=(device.type == "cuda")):
                losses = diffusion([gt, inp, task])
                loss = sum(losses)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(diffusion.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            ema.update()

            global_step += 1
            total_loss += float(loss.item())

        avg_loss = total_loss / max(1, len(loader))
        elapsed = time.time() - t0
        print(f"[Epoch {epoch}] loss={avg_loss:.6f} time={elapsed:.1f}s")

        log_path = log_dir / "train_log.csv"
        new_file = not log_path.exists()
        with open(log_path, "a", encoding="utf-8") as f:
            if new_file:
                f.write("epoch,step,loss,time_sec\n")
            f.write(f"{epoch},{global_step},{avg_loss:.8f},{elapsed:.2f}\n")

        if epoch % args.save_every == 0 or epoch == args.epochs:
            path = save_dir / f"model-{epoch}.pt"
            torch.save({
                "step": global_step,
                "model": diffusion.state_dict(),
                "opt0": optimizer.state_dict(),
                "ema": ema.state_dict(),
                "scaler": scaler.state_dict() if device.type == "cuda" else None,
                "protocol": {
                    "method": "DiffUIR-B",
                    "training_data": "Same all-in-one mixture as VTN",
                    "max_per_task": args.max_per_task,
                    "image_size": image_size,
                    "sampling_timesteps": args.sampling_timesteps,
                },
            }, path)
            print(f"[Saved] {path}")


if __name__ == "__main__":
    main()
