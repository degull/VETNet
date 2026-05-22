import argparse
import contextlib
import csv
import os
import sys
import time
from pathlib import Path

import torch

try:
    from ptflops import get_model_complexity_info
except Exception:
    get_model_complexity_info = None


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.restormer_volterra import RestormerVolterra


def build_model(name: str, rank: int):
    use_volterra = name == "VTN"
    return RestormerVolterra(
        in_channels=3,
        out_channels=3,
        dim=48,
        num_blocks=[4, 6, 6, 8],
        num_refinement_blocks=4,
        heads=[1, 2, 4, 8],
        ffn_expansion_factor=2.66,
        bias=False,
        LayerNorm_type="WithBias",
        volterra_rank=rank,
        use_volterra_mdta=use_volterra,
        use_volterra_gdfn=use_volterra,
    )


def count_params(model):
    return sum(p.numel() for p in model.parameters()) / 1e6


def measure_flops(model, size):
    if get_model_complexity_info is None:
        return None
    try:
        macs, _ = get_model_complexity_info(
            model,
            (3, size, size),
            as_strings=False,
            print_per_layer_stat=False,
            verbose=False,
        )
        return (macs or 0.0) / 1e9
    except Exception as exc:
        print(f"[WARN] FLOPs failed: {type(exc).__name__}: {exc}")
        return None


def benchmark_model(name, rank, size, warmup, iters, device):
    model = build_model(name, rank).to(device).eval()
    use_amp = device.type == "cuda"
    dummy = torch.randn(1, 3, size, size, device=device)

    params = count_params(model)
    flops = measure_flops(model, size)

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    ctx = torch.autocast(device_type="cuda", dtype=torch.float16) if use_amp else contextlib.nullcontext()
    with torch.no_grad():
        with ctx:
            for _ in range(warmup):
                _ = model(dummy)
        if device.type == "cuda":
            torch.cuda.synchronize()

        start = time.time()
        with ctx:
            for _ in range(iters):
                _ = model(dummy)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = (time.time() - start) / iters

    vram = torch.cuda.max_memory_allocated(device) / 1024**2 if device.type == "cuda" else 0.0
    return {
        "Method": name,
        "Input": f"1x3x{size}x{size}",
        "Params(M)": f"{params:.2f}",
        "FLOPs(G)": "-" if flops is None else f"{flops:.2f}",
        "Time(ms)": f"{elapsed * 1000:.2f}",
        "Iter/s": f"{1.0 / elapsed:.2f}",
        "VRAM(MB)": f"{vram:.2f}",
    }


def write_outputs(rows):
    out_dir = ROOT / "experiments"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "table2_efficiency.csv"
    md_path = out_dir / "table2_efficiency.md"

    headers = list(rows[0].keys())
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)

    with md_path.open("w", encoding="utf-8") as f:
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("|" + "|".join(["---"] * len(headers)) + "|\n")
        for row in rows:
            f.write("| " + " | ".join(row[h] for h in headers) + " |\n")

    print(f"[Saved] {csv_path}")
    print(f"[Saved] {md_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    print(f"[Device] {device}")
    print(f"[Input] 1x3x{args.size}x{args.size}")
    rows = []
    for name in ["Restormer", "VTN"]:
        print(f"[Benchmark] {name}")
        rows.append(benchmark_model(name, args.rank, args.size, args.warmup, args.iters, device))

    print("\n| Method | Input | Params(M) | FLOPs(G) | Time(ms) | Iter/s | VRAM(MB) |")
    print("|---|---|---:|---:|---:|---:|---:|")
    for row in rows:
        print(
            f"| {row['Method']} | {row['Input']} | {row['Params(M)']} | {row['FLOPs(G)']} | "
            f"{row['Time(ms)']} | {row['Iter/s']} | {row['VRAM(MB)']} |"
        )

    write_outputs(rows)


if __name__ == "__main__":
    main()
