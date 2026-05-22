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
WORKSPACE_DIR = ROOT_DIR.parent
ADAIR_DIR = WORKSPACE_DIR / "baselines" / "AdaIR"

for p in [ROOT_DIR, ADAIR_DIR]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from config import CHECKPOINT_ROOT, result_dir
from net.model import AdaIR
from train_p2_restormer_baseline import build_train_dataset, scheduled_size


def parse_args():
    parser = argparse.ArgumentParser(description="Train a fair P2 all-in-one AdaIR baseline.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--start-epoch", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max-per-task", type=int, default=160,
                        help="Samples per task per epoch. Default 160 matches VTN unified_balanced160.")
    parser.add_argument("--save-name", default="adair_p2_all_in_one")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--amp", action="store_true",
                        help="Enable mixed precision. Off by default because AdaIR FFT fails on non-power-of-two feature sizes in fp16.")
    return parser.parse_args()


def main():
    args = parse_args()
    if not ADAIR_DIR.exists():
        raise FileNotFoundError(f"AdaIR repo was not found: {ADAIR_DIR}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = CHECKPOINT_ROOT / args.save_name
    save_dir.mkdir(parents=True, exist_ok=True)
    log_dir = result_dir(f"train_{args.save_name}")

    model = AdaIR(decoder=True).to(device)
    if args.resume:
        state = torch.load(args.resume, map_location=device)
        model.load_state_dict(state, strict=True)
        print(f"[Resume] {args.resume}")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.L1Loss()
    use_amp = args.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    print(f"[Device] {device}")
    print(f"[Save] {save_dir}")
    print("[Protocol] P2 all-in-one AdaIR baseline.")
    print("[Protocol] Official AdaIR architecture, retrained from scratch.")
    print("[Protocol] Matches the revised Table 1 protocol: Rain100H, Rain100L, GoPro, RESIDE-6K, CSD.")
    print("[Protocol] Resize schedule: epoch 1-29=128, 30-59=192, 60+=256.")
    print(f"[Protocol] AMP enabled: {use_amp}")

    with open(log_dir / "protocol.txt", "w", encoding="utf-8") as f:
        f.write("P2 all-in-one AdaIR baseline for Table 1\n")
        f.write("Direct comparison target: VTN --ckpt-key unified_balanced160\n")
        f.write("Architecture: official AdaIR, decoder=True\n")
        f.write("Initialization: from scratch\n")
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

        for inp, gt in tqdm(loader, desc=f"Epoch {epoch}/{args.epochs}"):
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


if __name__ == "__main__":
    main()
