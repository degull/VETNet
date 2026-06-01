import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import CHECKPOINTS, DATA
from models.restormer_volterra import RestormerVolterra
from models.volterra_layer import VolterraLayer2D


def load_rgb(path, size=None):
    img = Image.open(path).convert("RGB")
    if size:
        img.thumbnail((size, size), Image.Resampling.LANCZOS)
    arr = np.asarray(img).astype(np.float32) / 255.0
    ten = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return img, arr, ten


def to_numpy_img(tensor):
    arr = tensor.detach().float().clamp(0, 1).squeeze(0).permute(1, 2, 0).cpu().numpy()
    return arr


def to_pil(arr):
    return Image.fromarray((np.clip(arr, 0, 1) * 255.0 + 0.5).astype(np.uint8))


def load_checkpoint(model, ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt.get("state_dict", ckpt.get("model", ckpt))
    clean = {}
    for key, value in state.items():
        key = key.replace("module.", "")
        clean[key] = value
    missing, unexpected = model.load_state_dict(clean, strict=False)
    if missing:
        print(f"[Warn] missing keys: {len(missing)}")
    if unexpected:
        print(f"[Warn] unexpected keys: {len(unexpected)}")
    return model


def infer(model, inp, tile=0):
    with torch.no_grad():
        if tile <= 0:
            return model(inp).clamp(0, 1)

        b, c, h, w = inp.shape
        out = torch.zeros_like(inp)
        weight = torch.zeros_like(inp)
        stride = tile // 2
        for y in range(0, h, stride):
            for x in range(0, w, stride):
                y0 = min(y, max(h - tile, 0))
                x0 = min(x, max(w - tile, 0))
                patch = inp[:, :, y0 : y0 + tile, x0 : x0 + tile]
                pred = model(patch).clamp(0, 1)
                out[:, :, y0 : y0 + tile, x0 : x0 + tile] += pred
                weight[:, :, y0 : y0 + tile, x0 : x0 + tile] += 1
        return out / weight.clamp_min(1)


class VolterraHook:
    def __init__(self, model):
        self.records = {"mdta": [], "gdfn": []}
        self.handles = []
        for name, module in model.named_modules():
            if not isinstance(module, VolterraLayer2D):
                continue
            key = None
            if "volterra1" in name:
                key = "mdta"
            elif "volterra2" in name:
                key = "gdfn"
            if key is None:
                continue
            self.handles.append(module.register_forward_hook(self._make_hook(key)))

    def _make_hook(self, key):
        def hook(_module, _inputs, output):
            self.records[key].append(output.detach().float().cpu())

        return hook

    def close(self):
        for handle in self.handles:
            handle.remove()


def activation_map(tensors, target_hw, mode="l2"):
    if not tensors:
        return np.zeros(target_hw, dtype=np.float32)

    maps = []
    for feat in tensors:
        if mode == "mean_abs":
            amap = feat.abs().mean(dim=1, keepdim=True)
        else:
            amap = torch.sqrt((feat * feat).sum(dim=1, keepdim=True) + 1e-8)
        amap = F.interpolate(amap, size=target_hw, mode="bilinear", align_corners=False)
        amap = amap.squeeze().numpy()
        lo, hi = np.percentile(amap, [2, 98])
        amap = (amap - lo) / max(hi - lo, 1e-6)
        maps.append(np.clip(amap, 0, 1))
    return np.mean(maps, axis=0).astype(np.float32)


def colorize_heatmap(gray):
    gray = np.clip(gray, 0, 1)
    try:
        import matplotlib

        return matplotlib.colormaps["turbo"](gray)[..., :3].astype(np.float32)
    except Exception:
        r = np.clip(1.5 * gray, 0, 1)
        g = np.clip(1.5 - np.abs(gray - 0.5) * 3.0, 0, 1)
        b = np.clip(1.5 * (1.0 - gray), 0, 1)
        return np.stack([r, g, b], axis=-1).astype(np.float32)


def overlay(base, gray, alpha=0.45):
    heat = colorize_heatmap(gray)
    return np.clip((1 - alpha) * base + alpha * heat, 0, 1)


def error_gray(pred, gt):
    err = np.abs(pred - gt).mean(axis=2)
    lo, hi = np.percentile(err, [2, 98])
    return np.clip((err - lo) / max(hi - lo, 1e-6), 0, 1)


def auto_crop_box(input_arr, gt_arr, crop=96):
    h, w = input_arr.shape[:2]
    crop = min(crop, h, w)
    err = np.abs(input_arr - gt_arr).mean(axis=2)
    stride = max(crop // 4, 8)
    best = (0.0, 0, 0)
    for y in range(0, max(h - crop + 1, 1), stride):
        for x in range(0, max(w - crop + 1, 1), stride):
            score = float(err[y : y + crop, x : x + crop].mean())
            if score > best[0]:
                best = (score, x, y)
    _, x, y = best
    return x, y, x + crop, y + crop


def crop_and_resize(arr, box, out_size):
    img = to_pil(arr).crop(box).resize((out_size, out_size), Image.Resampling.BICUBIC)
    return np.asarray(img).astype(np.float32) / 255.0


def add_label(img, label):
    pad = 30
    out = Image.new("RGB", (img.width, img.height + pad), "white")
    out.paste(img, (0, pad))
    draw = ImageDraw.Draw(out)
    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except Exception:
        font = ImageFont.load_default()
    draw.text((8, 6), label, fill=(0, 0, 0), font=font)
    return out


def make_grid(images, labels, cols):
    labeled = [add_label(to_pil(img), lab) for img, lab in zip(images, labels)]
    cell_w = max(img.width for img in labeled)
    cell_h = max(img.height for img in labeled)
    rows = math.ceil(len(labeled) / cols)
    grid = Image.new("RGB", (cols * cell_w, rows * cell_h), "white")
    for i, img in enumerate(labeled):
        x = (i % cols) * cell_w
        y = (i // cols) * cell_h
        grid.paste(img, (x, y))
    return grid


def default_sample():
    inp = DATA["csd"] / "Test" / "Snow" / "111.tif"
    gt = DATA["csd"] / "Test" / "Gt" / "111.tif"
    return inp, gt


def resolve_ckpt(value):
    if value in CHECKPOINTS:
        return CHECKPOINTS[value]
    return Path(value)


def build_model(kind, ckpt, device):
    if kind == "backbone":
        model = RestormerVolterra(use_volterra_mdta=False, use_volterra_gdfn=False).to(device)
    else:
        model = RestormerVolterra().to(device)
    load_checkpoint(model, ckpt, device)
    model.eval()
    return model


def write_latex(out_dir):
    snippet = r"""\begin{figure*}[t]
\centering
\includegraphics[width=\linewidth]{figures/figure_qualitative_error_interaction.png}
\caption{Qualitative and interpretability visualization. The first two rows compare degraded input, a Transformer backbone, VTN, and ground truth with zoomed crops. The last row shows restoration error maps and Volterra interaction maps from the MDTA and GDFN pathways. Brighter interaction regions indicate stronger second-order Volterra responses, which are concentrated near degradation-sensitive edges, streaks, and texture boundaries.}
\label{fig:volterra_visualization}
\end{figure*}
"""
    (out_dir / "paper_latex_snippet.tex").write_text(snippet, encoding="utf-8")


def first_existing(paths, limit):
    out = []
    for path in paths:
        if path.exists():
            out.append(path)
        if len(out) >= limit:
            break
    return out


def resolve_gopro_path(path_text):
    path_text = path_text.replace("\\", "/")
    anchor = "GOPRO_Large/"
    if anchor in path_text:
        return DATA["gopro"] / path_text.split(anchor, 1)[1]
    return Path(path_text)


def gather_tsne_inputs(per_task):
    samples = []
    rain = sorted((DATA["rain100h"] / "test" / "rain").glob("*"))
    for path in first_existing(rain, per_task):
        samples.append(("Rain", path))

    csv_path = DATA["gopro"] / "gopro_test_pairs.csv"
    if csv_path.exists():
        with csv_path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                path = resolve_gopro_path(row.get("dist_img", ""))
                if path.exists():
                    samples.append(("Blur", path))
                if sum(1 for label, _ in samples if label == "Blur") >= per_task:
                    break

    hazy = sorted((DATA["reside"] / "test" / "hazy").glob("*"))
    for path in first_existing(hazy, per_task):
        samples.append(("Haze", path))

    snow = sorted((DATA["csd"] / "Test" / "Snow").glob("*"))
    for path in first_existing(snow, per_task):
        samples.append(("Snow", path))

    return samples


def extract_embedding(model, image_path, device, size):
    _pil, _arr, inp = load_rgb(image_path, size)
    inp = inp.to(device)
    captures = []

    def hook(_module, _inputs, output):
        captures.append(output.detach().float().cpu())

    handle = model.refinement.register_forward_hook(hook)
    with torch.no_grad():
        _ = model(inp)
    handle.remove()
    feat = captures[-1] if captures else _.detach().float().cpu()
    return feat.mean(dim=(2, 3)).squeeze(0).numpy()


def reduce_to_2d(features):
    x = np.asarray(features, dtype=np.float32)
    x = x - x.mean(axis=0, keepdims=True)
    try:
        from sklearn.manifold import TSNE

        perplexity = max(2, min(15, len(x) // 3))
        return TSNE(n_components=2, perplexity=perplexity, init="pca", learning_rate="auto", random_state=0).fit_transform(x)
    except Exception:
        u, s, _vh = np.linalg.svd(x, full_matrices=False)
        return u[:, :2] * s[:2]


def make_tsne_figure(model, device, out_dir, per_task=20, size=128):
    samples = gather_tsne_inputs(per_task)
    if len(samples) < 4:
        print("[Warn] not enough samples for t-SNE")
        return None

    labels = []
    feats = []
    for label, path in samples:
        try:
            feats.append(extract_embedding(model, path, device, size))
            labels.append(label)
        except Exception as exc:
            print(f"[Warn] skip {path}: {exc}")

    emb = reduce_to_2d(feats)
    try:
        import matplotlib.pyplot as plt

        colors = {"Rain": "#1f77b4", "Blur": "#ff7f0e", "Haze": "#2ca02c", "Snow": "#d62728"}
        plt.figure(figsize=(5.0, 4.0), dpi=220)
        for label in ["Rain", "Blur", "Haze", "Snow"]:
            idx = [i for i, item in enumerate(labels) if item == label]
            if not idx:
                continue
            plt.scatter(emb[idx, 0], emb[idx, 1], s=24, label=label, color=colors[label], alpha=0.85)
        plt.xticks([])
        plt.yticks([])
        plt.legend(frameon=False, loc="best")
        plt.tight_layout()
        path = out_dir / "figure_feature_tsne.png"
        plt.savefig(path, bbox_inches="tight")
        plt.close()
        print(f"[Saved] {path}")
        return path
    except Exception as exc:
        print(f"[Warn] could not draw t-SNE: {exc}")
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--gt", type=Path, default=None)
    parser.add_argument("--vtn-ckpt", default="unified_balanced160")
    parser.add_argument("--baseline-ckpt", type=Path, default=ROOT / "checkpoints" / "table5_backbone" / "epoch_100_loss0.02768.pth")
    parser.add_argument("--baseline-kind", choices=["backbone", "vtn"], default="backbone")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results" / "paper_visualizations" / "example")
    parser.add_argument("--size", type=int, default=384)
    parser.add_argument("--crop", type=int, default=96)
    parser.add_argument("--tile", type=int, default=0)
    parser.add_argument("--map-mode", choices=["l2", "mean_abs"], default="l2")
    parser.add_argument("--make-tsne", action="store_true")
    parser.add_argument("--tsne-per-task", type=int, default=20)
    args = parser.parse_args()

    if args.input is None or args.gt is None:
        args.input, args.gt = default_sample()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")
    print(f"[Input] {args.input}")
    print(f"[GT] {args.gt}")

    _pil, input_arr, inp = load_rgb(args.input, args.size)
    _gt_pil, gt_arr, _ = load_rgb(args.gt, args.size)
    inp = inp.to(device)

    vtn_ckpt = resolve_ckpt(args.vtn_ckpt)
    vtn = build_model("vtn", vtn_ckpt, device)
    hooks = VolterraHook(vtn)
    vtn_pred_t = infer(vtn, inp, args.tile)
    hooks.close()
    vtn_pred = to_numpy_img(vtn_pred_t)

    baseline_pred = None
    if args.baseline_ckpt.exists():
        baseline = build_model(args.baseline_kind, args.baseline_ckpt, device)
        baseline_pred = to_numpy_img(infer(baseline, inp, args.tile))
    else:
        print(f"[Warn] baseline checkpoint not found: {args.baseline_ckpt}")
        baseline_pred = input_arr

    h, w = input_arr.shape[:2]
    mdta_map = activation_map(hooks.records["mdta"], (h, w), args.map_mode)
    gdfn_map = activation_map(hooks.records["gdfn"], (h, w), args.map_mode)
    baseline_err = error_gray(baseline_pred, gt_arr)
    vtn_err = error_gray(vtn_pred, gt_arr)

    box = auto_crop_box(input_arr, gt_arr, args.crop)
    zoom_size = min(180, max(input_arr.shape[:2]))
    rows = [
        input_arr,
        baseline_pred,
        vtn_pred,
        gt_arr,
        crop_and_resize(input_arr, box, zoom_size),
        crop_and_resize(baseline_pred, box, zoom_size),
        crop_and_resize(vtn_pred, box, zoom_size),
        crop_and_resize(gt_arr, box, zoom_size),
        colorize_heatmap(baseline_err),
        colorize_heatmap(vtn_err),
        overlay(input_arr, mdta_map),
        overlay(input_arr, gdfn_map),
    ]
    labels = [
        "Degraded",
        "Backbone",
        "VTN",
        "GT",
        "Zoom Input",
        "Zoom Backbone",
        "Zoom VTN",
        "Zoom GT",
        "Backbone Error",
        "VTN Error",
        "MDTA Volterra Map",
        "GDFN Volterra Map",
    ]
    figure = make_grid(rows, labels, cols=4)
    figure_path = args.out_dir / "figure_qualitative_error_interaction.png"
    figure.save(figure_path)

    to_pil(vtn_pred).save(args.out_dir / "vtn_restored.png")
    to_pil(baseline_pred).save(args.out_dir / "baseline_restored.png")
    to_pil(colorize_heatmap(baseline_err)).save(args.out_dir / "error_baseline.png")
    to_pil(colorize_heatmap(vtn_err)).save(args.out_dir / "error_vtn.png")
    to_pil(overlay(input_arr, mdta_map)).save(args.out_dir / "volterra_mdta_overlay.png")
    to_pil(overlay(input_arr, gdfn_map)).save(args.out_dir / "volterra_gdfn_overlay.png")
    write_latex(args.out_dir)

    if args.make_tsne:
        make_tsne_figure(vtn, device, args.out_dir, per_task=args.tsne_per_task)

    print(f"[Saved] {figure_path}")
    print(f"[Saved] {args.out_dir / 'paper_latex_snippet.tex'}")
    print("[Layout] Row1: degraded / backbone / VTN / GT")
    print("[Layout] Row2: zoom crops")
    print("[Layout] Row3: backbone error / VTN error / MDTA map / GDFN map")


if __name__ == "__main__":
    main()
