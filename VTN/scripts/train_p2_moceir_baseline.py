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
MOCEIR_SRC = WORKSPACE_DIR / "baselines" / "MoCE-IR" / "src"

for p in [ROOT_DIR, MOCEIR_SRC]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from config import CHECKPOINT_ROOT, result_dir
from net.moce_ir import MoCEIR
from train_p2_restormer_baseline import build_train_dataset, scheduled_size


def build_moceir(model_size: str):
    if model_size == "S":
        dim = 32
    elif model_size == "B":
        dim = 48
    else:
        raise ValueError("--model-size must be S or B")

    return MoCEIR(
        dim=dim,
        num_blocks=[4, 6, 6, 8],
        num_dec_blocks=[2, 4, 4],
        levels=4,
        heads=[1, 2, 4, 8],
        num_refinement_blocks=4,
        topk=1,
        num_experts=4,
        rank=2,
        with_complexity=True,
        depth_type="constant",
        stage_depth=[1, 1, 1],
        rank_type="spread",
        complexity_scale="max",
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Train a fair P2 all-in-one MoCE-IR baseline.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--start-epoch", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max-per-task", type=int, default=160,
                        help="Samples per task per epoch. Default 160 matches VTN unified_balanced160.")
    parser.add_argument("--save-name", default="moceir_s_p2_all_in_one")
    parser.add_argument("--model-size", choices=["S", "B"], default="S",
                        help="S=MoCE-IR-S lightweight, B=MoCE-IR base/heavy.")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--balance-loss-weight", type=float, default=0.01)
    parser.add_argument("--amp", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if not MOCEIR_SRC.exists():
        raise FileNotFoundError(f"MoCE-IR repo was not found: {MOCEIR_SRC}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = CHECKPOINT_ROOT / args.save_name
    save_dir.mkdir(parents=True, exist_ok=True)
    log_dir = result_dir(f"train_{args.save_name}")

    model = build_moceir(args.model_size).to(device)
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
    print(f"[Protocol] P2 all-in-one MoCE-IR-{args.model_size} baseline.")
    print("[Protocol] Official MoCE-IR architecture, retrained from scratch.")
    print("[Protocol] Matches the revised Table 1 protocol: Rain100H, Rain100L, GoPro, RESIDE-6K, CSD.")
    print("[Protocol] Resize schedule: epoch 1-29=128, 30-59=192, 60+=256.")
    print(f"[Protocol] Balance loss weight: {args.balance_loss_weight}")
    print(f"[Protocol] AMP enabled: {use_amp}")

    with open(log_dir / "protocol.txt", "w", encoding="utf-8") as f:
        f.write(f"P2 all-in-one MoCE-IR-{args.model_size} baseline for Table 1\n")
        f.write("Direct comparison target: VTN --ckpt-key unified_balanced160\n")
        f.write("Architecture: official MoCE-IR, with complexity routing\n")
        f.write("Initialization: from scratch\n")
        f.write("Tasks: Rain100H, Rain100L, GoPro, RESIDE-6K, CSD\n")
        f.write(f"Samples per task per epoch: {args.max_per_task}\n")
        f.write("Resize schedule: epoch 1-29=128, 30-59=192, 60+=256\n")
        f.write(f"Batch size: {args.batch_size}\n")
        f.write(f"LR: {args.lr}\n")
        f.write(f"Balance loss weight: {args.balance_loss_weight}\n")
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
        total_l1 = 0.0
        total_balance = 0.0
        print(f"[Epoch {epoch}] image_size={image_size} samples={len(dataset)}")

        for inp, gt in tqdm(loader, desc=f"Epoch {epoch}/{args.epochs}"):
            inp = inp.to(device, non_blocking=True)
            gt = gt.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                out = model(inp).clamp(0, 1)
                l1_loss = criterion(out, gt)
                balance_loss = model.total_loss if model.total_loss is not None else torch.zeros((), device=device)
                loss = l1_loss + args.balance_loss_weight * balance_loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += float(loss.item())
            total_l1 += float(l1_loss.item())
            total_balance += float(balance_loss.item())

        avg_loss = total_loss / max(1, len(loader))
        avg_l1 = total_l1 / max(1, len(loader))
        avg_balance = total_balance / max(1, len(loader))
        elapsed = time.time() - t0
        print(f"[Epoch {epoch}] loss={avg_loss:.6f} l1={avg_l1:.6f} balance={avg_balance:.6f} time={elapsed:.1f}s")

        log_path = log_dir / "train_log.csv"
        new_file = not log_path.exists()
        with open(log_path, "a", encoding="utf-8") as f:
            if new_file:
                f.write("epoch,loss,l1_loss,balance_loss,time_sec\n")
            f.write(f"{epoch},{avg_loss:.8f},{avg_l1:.8f},{avg_balance:.8f},{elapsed:.2f}\n")

        if epoch % args.save_every == 0 or epoch == args.epochs:
            path = save_dir / f"epoch_{epoch:03d}_loss{avg_loss:.5f}.pth"
            torch.save(model.state_dict(), path)
            print(f"[Saved] {path}")


if __name__ == "__main__":
    main()
