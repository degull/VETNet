import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

CUR_DIR = Path(__file__).resolve().parent
ROOT_DIR = CUR_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import CHECKPOINT_ROOT, result_dir
from train_p2_restormer_baseline import build_train_dataset, scheduled_size
from table5_ablation_common import VARIANTS, TABLE_ORDER, build_variant_model


def parse_args():
    parser = argparse.ArgumentParser(description="Train Table 5 ablation variants.")
    parser.add_argument("--variant", choices=TABLE_ORDER + ["all"], required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--start-epoch", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max-per-task", type=int, default=160)
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--resume", default=None,
                        help="Resume path. Only valid when training a single variant.")
    parser.add_argument("--amp", action="store_true")
    return parser.parse_args()


def train_one(variant: str, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = VARIANTS[variant]
    save_dir = CHECKPOINT_ROOT / cfg["save_name"]
    save_dir.mkdir(parents=True, exist_ok=True)
    log_dir = result_dir(f"train_{cfg['save_name']}")

    model = build_variant_model(variant).to(device)
    if args.resume:
        state = torch.load(args.resume, map_location=device)
        model.load_state_dict(state, strict=True)
        print(f"[Resume] {args.resume}")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.L1Loss()
    use_amp = args.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    print(f"[Variant] {variant}: {cfg['label']}")
    print(f"[Save] {save_dir}")
    print(f"[Protocol] Table 5 ablation, 5-task mixture: Rain100H, Rain100L, GoPro, RESIDE-6K, CSD.")
    print(f"[Config] {cfg['model_kwargs']}")
    print(f"[AMP] {use_amp}")

    with open(log_dir / "protocol.txt", "w", encoding="utf-8") as f:
        f.write(f"Table 5 ablation: {variant} / {cfg['label']}\n")
        f.write(f"Model kwargs: {cfg['model_kwargs']}\n")
        f.write("Tasks: Rain100H, Rain100L, GoPro, RESIDE-6K, CSD\n")
        f.write(f"Samples per task per epoch: {args.max_per_task}\n")
        f.write("Resize schedule: epoch 1-29=128, 30-59=192, 60+=256\n")
        f.write(f"Batch size: {args.batch_size}\n")
        f.write(f"LR: {args.lr}\n")
        f.write(f"AMP: {use_amp}\n")

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

        model.train()
        t0 = time.time()
        total_loss = 0.0
        print(f"[Epoch {epoch}] image_size={image_size} samples={len(dataset)}")

        for inp, gt in tqdm(loader, desc=f"{variant} {epoch}/{args.epochs}"):
            inp = inp.to(device, non_blocking=True)
            gt = gt.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                out = model(inp).clamp(0, 1)
                loss = criterion(out, gt)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.item())

        avg_loss = total_loss / max(1, len(loader))
        elapsed = time.time() - t0
        print(f"[Epoch {epoch}] loss={avg_loss:.6f} time={elapsed:.1f}s")

        log_path = log_dir / "train_log.csv"
        new_file = not log_path.exists()
        with open(log_path, "a", encoding="utf-8") as f:
            if new_file:
                f.write("epoch,loss,time_sec\n")
            f.write(f"{epoch},{avg_loss:.8f},{elapsed:.2f}\n")

        if epoch % args.save_every == 0 or epoch == args.epochs:
            path = save_dir / f"epoch_{epoch:03d}_loss{avg_loss:.5f}.pth"
            torch.save(model.state_dict(), path)
            print(f"[Saved] {path}")


def main():
    args = parse_args()
    variants = TABLE_ORDER if args.variant == "all" else [args.variant]
    if args.variant == "all" and args.resume:
        raise ValueError("--resume can only be used with one variant")
    for variant in variants:
        train_one(variant, args)


if __name__ == "__main__":
    main()
