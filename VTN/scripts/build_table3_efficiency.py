import argparse
import csv
import importlib
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

CUR_DIR = Path(__file__).resolve().parent
ROOT_DIR = CUR_DIR.parent
WORKSPACE_DIR = ROOT_DIR.parent

for p in [ROOT_DIR]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from models.restormer_volterra import RestormerVolterra


AVG_PSNR = {
    "Restormer": "27.91",
    "PromptIR": "27.68",
    "DiffUIR": "25.55",
    "DA-CLIP": "-",
    "AdaIR": "26.93",
    "MoCE-IR": "27.99",
    "MambaIRv2": "-",
    "HINT": "27.22",
    "VTN": "28.16",
}


def clear_modules(prefixes):
    for name in list(sys.modules.keys()):
        if any(name == p or name.startswith(p + ".") for p in prefixes):
            del sys.modules[name]


def prepend(path: Path):
    s = str(path)
    if s in sys.path:
        sys.path.remove(s)
    sys.path.insert(0, s)


def build_restormer():
    return RestormerVolterra(use_volterra_mdta=False, use_volterra_gdfn=False)


def build_vtn():
    return RestormerVolterra()


def build_promptir():
    clear_modules(["net"])
    prepend(WORKSPACE_DIR / "baselines" / "PromptIR")
    from net.model import PromptIR
    return PromptIR(decoder=True)


def build_adair():
    clear_modules(["net"])
    prepend(WORKSPACE_DIR / "baselines" / "AdaIR")
    from net.model import AdaIR
    return AdaIR(decoder=True)


def build_moceir():
    clear_modules(["net"])
    prepend(WORKSPACE_DIR / "baselines" / "MoCE-IR" / "src")
    from net.moce_ir import MoCEIR
    return MoCEIR(
        dim=32,
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


def build_hint():
    clear_modules(["basicsr"])
    prepend(WORKSPACE_DIR / "baselines" / "HINT")
    from basicsr.models.archs.HINT_arch import HINT
    return HINT(dim=32, num_blocks=[2, 3, 3, 4], num_refinement_blocks=2, heads=[1, 2, 4, 8])


class DiffUIRInferenceWrapper(nn.Module):
    def __init__(self, sampling_timesteps=3):
        super().__init__()
        clear_modules(["src"])
        diffuir_root = WORKSPACE_DIR / "baselines" / "DiffUIR"
        diffuir_deps = ROOT_DIR / ".deps_diffuir"
        if diffuir_deps.exists():
            prepend(diffuir_deps)
        prepend(diffuir_root)
        from src.model import ResidualDiffusion, UnetRes

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
        self.diffusion = ResidualDiffusion(
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

    def forward(self, x):
        out = self.diffusion.sample(x_input=x, batch_size=x.shape[0], last=True)
        if isinstance(out, list):
            return out[-1]
        return out


class DiffUIRStepWrapper(nn.Module):
    def __init__(self, diffusion_wrapper: DiffUIRInferenceWrapper):
        super().__init__()
        self.model = diffusion_wrapper.diffusion.model

    def forward(self, x):
        time = [
            torch.ones(x.shape[0], device=x.device) * 500.0,
            torch.ones(x.shape[0], device=x.device) * 500.0,
        ]
        return self.model(torch.cat([x, x], dim=1), time)[0]


def build_diffuir():
    return DiffUIRInferenceWrapper(sampling_timesteps=3)


def build_mambairv2():
    clear_modules(["basicsr"])
    prepend(WORKSPACE_DIR / "baselines" / "MambaIR")
    from basicsr.archs.mambairv2_arch import MambaIRv2
    return MambaIRv2(
        img_size=64,
        patch_size=1,
        in_chans=3,
        embed_dim=48,
        d_state=8,
        depths=(5, 5, 5, 5),
        num_heads=(4, 4, 4, 4),
        window_size=16,
        inner_rank=32,
        num_tokens=64,
        convffn_kernel_size=5,
        mlp_ratio=1.0,
        patch_norm=True,
        use_checkpoint=False,
        upscale=1,
        img_range=1.0,
        upsampler="",
        resi_connection="1conv",
    )


BUILDERS = {
    "Restormer": build_restormer,
    "PromptIR": build_promptir,
    "DiffUIR": build_diffuir,
    "DA-CLIP": None,
    "AdaIR": build_adair,
    "MoCE-IR": build_moceir,
    "MambaIRv2": build_mambairv2,
    "HINT": build_hint,
    "VTN": build_vtn,
}


def count_params(model):
    return sum(p.numel() for p in model.parameters()) / 1e6


def measure_flops(model, x, method):
    try:
        from thop import profile
        flops_model = model
        flops_x = x
        multiplier = 1
        if method == "DiffUIR":
            flops_model = DiffUIRStepWrapper(model).to(x.device).eval()
            multiplier = model.diffusion.sampling_timesteps
        with torch.no_grad():
            macs, _ = profile(flops_model, inputs=(flops_x,), verbose=False)
        # Most restoration papers report MACs as FLOPs-like cost; keep the value in G.
        return macs * multiplier / 1e9
    except Exception as exc:
        return None


@torch.no_grad()
def measure_runtime_memory(model, x, warmup, repeats, amp):
    device = x.device
    model.eval()

    use_cuda = device.type == "cuda"
    if use_cuda:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    for _ in range(warmup):
        with torch.amp.autocast(device_type="cuda", enabled=(amp and use_cuda)):
            y = model(x)
        if isinstance(y, (list, tuple)):
            y = y[-1]

    if use_cuda:
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(repeats):
            with torch.amp.autocast(device_type="cuda", enabled=(amp and use_cuda)):
                y = model(x)
            if isinstance(y, (list, tuple)):
                y = y[-1]
        end.record()
        torch.cuda.synchronize()
        latency_ms = start.elapsed_time(end) / repeats
        memory_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    else:
        t0 = time.perf_counter()
        for _ in range(repeats):
            y = model(x)
            if isinstance(y, (list, tuple)):
                y = y[-1]
        latency_ms = (time.perf_counter() - t0) * 1000.0 / repeats
        memory_mb = None

    return latency_ms, memory_mb


def fmt_params(v):
    return "-" if v is None else f"{v:.2f}M"


def fmt_flops(v):
    return "-" if v is None else f"{v:.2f}G"


def fmt_latency(v):
    return "-" if v is None else f"{v:.2f} ms"


def fmt_memory(v):
    return "-" if v is None else f"{v:.0f} MB"


def parse_args():
    parser = argparse.ArgumentParser(description="Build Table 3 efficiency comparison.")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--amp", action="store_true", help="Measure latency/memory with AMP.")
    parser.add_argument("--methods", nargs="*", default=list(BUILDERS.keys()))
    parser.add_argument("--out-csv", default=str(ROOT_DIR / "experiments" / "table3_efficiency.csv"))
    parser.add_argument("--out-md", default=str(ROOT_DIR / "experiments" / "table3_efficiency.md"))
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.randn(1, 3, args.image_size, args.image_size, device=device)
    rows = []

    print(f"[Device] {device}")
    print(f"[Input ] 1x3x{args.image_size}x{args.image_size}")
    print(f"[Timing] warmup={args.warmup}, repeats={args.repeats}, AMP={args.amp}")

    for method in args.methods:
        builder = BUILDERS.get(method)
        row = {
            "Method": method,
            "Params": "-",
            "FLOPs": "-",
            "Latency": "-",
            "Memory": "-",
            "Avg PSNR": AVG_PSNR.get(method, "-"),
            "Note": "",
        }
        if builder is None:
            row["Note"] = "not measured"
            rows.append(row)
            print(f"[Skip] {method}: no local runnable builder")
            continue

        try:
            model = builder().to(device).eval()
            params = count_params(model)
            flops = measure_flops(model, x, method)
            latency, memory = measure_runtime_memory(model, x, args.warmup, args.repeats, args.amp)
            row.update({
                "Params": fmt_params(params),
                "FLOPs": fmt_flops(flops),
                "Latency": fmt_latency(latency),
                "Memory": fmt_memory(memory),
            })
            print(f"[OK] {method}: {row['Params']}, {row['FLOPs']}, {row['Latency']}, {row['Memory']}")
        except Exception as exc:
            row["Note"] = f"failed: {type(exc).__name__}: {exc}"
            print(f"[Fail] {method}: {row['Note']}")
        finally:
            try:
                del model
            except Exception:
                pass
            if device.type == "cuda":
                torch.cuda.empty_cache()

        rows.append(row)

    out_csv = Path(args.out_csv)
    out_md = Path(args.out_md)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = ["Method", "Params", "FLOPs", "Latency", "Memory", "Avg PSNR", "Note"]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    with open(out_md, "w", encoding="utf-8") as f:
        f.write("| Method | Params | FLOPs | Latency | Memory | Avg PSNR |\n")
        f.write("|---|---:|---:|---:|---:|---:|\n")
        for row in rows:
            f.write(
                f"| {row['Method']} | {row['Params']} | {row['FLOPs']} | {row['Latency']} | "
                f"{row['Memory']} | {row['Avg PSNR']} |\n"
            )

    print(f"[Table 3 CSV] {out_csv}")
    print(f"[Table 3 MD ] {out_md}")


if __name__ == "__main__":
    main()
