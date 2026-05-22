import argparse
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import ConcatDataset, DataLoader
from torchvision import transforms
from tqdm import tqdm

CUR_DIR = Path(__file__).resolve().parent
ROOT_DIR = CUR_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import DATA, CHECKPOINT_ROOT, result_dir, as_str
from datasets.rain100h_dataset import Rain100HDataset
from datasets.rain100l_dataset import Rain100LDataset
from datasets.gopro_dataset import GoProDataset
from datasets.reside_dataset import RESIDEDataset
from datasets.csd_dataset import CSDDataset
from models.restormer_volterra import RestormerVolterra


RESIZE_SCHEDULE = {0: 128, 30: 192, 60: 256}


def scheduled_size(epoch: int) -> int:
    return max(size for start_epoch, size in RESIZE_SCHEDULE.items() if epoch >= start_epoch)


def build_train_dataset(image_size: int, max_per_task: int | None = None):
    tfm = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])

    datasets = [
        Rain100HDataset(as_str(DATA["rain100h"] / "train"), transform=tfm),
        Rain100LDataset(as_str(DATA["rain100l"] / "train"), transform=tfm),
        GoProDataset(as_str(DATA["gopro"] / "gopro_train_pairs.csv"), transform=tfm),
        RESIDEDataset(root_dir=as_str(DATA["reside"]), split="train", transform=tfm, strict=True),
        CSDDataset(root_dir=as_str(DATA["csd"]), split="train", transform=tfm),
    ]

    if max_per_task is not None:
        datasets = [
            torch.utils.data.Subset(ds, torch.randperm(len(ds))[:min(max_per_task, len(ds))].tolist())
            for ds in datasets
        ]

    return ConcatDataset(datasets)


def parse_args():
    parser = argparse.ArgumentParser(description="Train a fair P2 all-in-one Restormer baseline.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--start-epoch", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max-per-task", type=int, default=160,
                        help="Samples per task per epoch. Default 160 matches VTN unified_balanced160.")
    parser.add_argument("--save-name", default="restormer_p2_all_in_one")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--save-every", type=int, default=1)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = CHECKPOINT_ROOT / args.save_name
    save_dir.mkdir(parents=True, exist_ok=True)
    log_dir = result_dir(f"train_{args.save_name}")

    model = RestormerVolterra(use_volterra_mdta=False, use_volterra_gdfn=False).to(device)
    if args.resume:
        state = torch.load(args.resume, map_location=device)
        model.load_state_dict(state, strict=True)
        print(f"[Resume] {args.resume}")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.L1Loss()
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    print(f"[Device] {device}")
    print(f"[Save] {save_dir}")
    print("[Protocol] P2 all-in-one Restormer baseline.")
    print("[Protocol] Volterra disabled in MDTA and FFN.")
    print("[Protocol] Matches the revised Table 1 protocol: Rain100H, Rain100L, GoPro, RESIDE-6K, CSD.")
    print("[Protocol] Resize schedule: epoch 1-29=128, 30-59=192, 60+=256.")

    with open(log_dir / "protocol.txt", "w", encoding="utf-8") as f:
        f.write("P2 all-in-one Restormer baseline for Table 1\n")
        f.write("Direct comparison target: VTN --ckpt-key unified_balanced160\n")
        f.write("Volterra MDTA: off\n")
        f.write("Volterra FFN: off\n")
        f.write("Tasks: Rain100H, Rain100L, GoPro, RESIDE-6K, CSD\n")
        f.write(f"Samples per task per epoch: {args.max_per_task}\n")
        f.write("Resize schedule: epoch 1-29=128, 30-59=192, 60+=256\n")
        f.write(f"Batch size: {args.batch_size}\n")
        f.write(f"LR: {args.lr}\n")

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
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                out = model(inp)
                loss = criterion(out, gt)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.item())

        avg_loss = total_loss / max(1, len(loader))
        elapsed = time.time() - t0
        print(f"[Epoch {epoch}] loss={avg_loss:.6f} time={elapsed:.1f}s")

        with open(log_dir / "train_log.csv", "a", encoding="utf-8") as f:
            if epoch == 1 and f.tell() == 0:
                f.write("epoch,loss,time_sec\n")
            f.write(f"{epoch},{avg_loss:.8f},{elapsed:.2f}\n")

        if epoch % args.save_every == 0 or epoch == args.epochs:
            path = save_dir / f"epoch_{epoch:03d}_loss{avg_loss:.5f}.pth"
            torch.save(model.state_dict(), path)
            print(f"[Saved] {path}")


if __name__ == "__main__":
    main()
