import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from tqdm import tqdm

CUR_DIR = Path(__file__).resolve().parent
ROOT_DIR = CUR_DIR.parent
WORKSPACE_DIR = ROOT_DIR.parent
DACLP_ROOT = WORKSPACE_DIR / "baselines" / "DA-CLIP"
UIR_ROOT = DACLP_ROOT / "universal-image-restoration"
DACLP_CFG = UIR_ROOT / "config" / "daclip-sde"
THIRD_PARTY_EMA = ROOT_DIR / "third_party_ema"

for p in [ROOT_DIR, DACLP_CFG, UIR_ROOT, THIRD_PARTY_EMA]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from config import DATA, result_dir, as_str
from datasets.rain100h_dataset import Rain100HDataset
from datasets.rain100l_dataset import Rain100LDataset
from datasets.gopro_dataset import GoProDataset
from datasets.reside_dataset import RESIDEDataset
from datasets.csd_dataset import CSDDataset

import options as option
from models import create_model
import open_clip
import utils as daclip_utils
from data import util as data_util


EVAL_SIZE = 256
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ClipWrappedDataset(Dataset):
    def __init__(self, base):
        self.base = base

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        lq, gt = self.base[idx]
        lq_np = lq.permute(1, 2, 0).numpy()
        lq_clip = data_util.clip_transform(lq_np)
        return {"LQ": lq, "GT": gt, "LQ_clip": lq_clip}


def tensor_to_hwc01(x):
    if x.dim() == 4:
        x = x[0]
    return x.detach().float().clamp(0, 1).cpu().numpy().transpose(1, 2, 0)


def compute_metrics(pred, gt):
    pred_np = tensor_to_hwc01(pred)
    gt_np = tensor_to_hwc01(gt)
    return (
        peak_signal_noise_ratio(gt_np, pred_np, data_range=1.0),
        structural_similarity(gt_np, pred_np, channel_axis=2, data_range=1.0),
    )


def build_base_dataset(name):
    tfm = transforms.Compose([transforms.Resize((EVAL_SIZE, EVAL_SIZE)), transforms.ToTensor()])
    if name == "Rain100H":
        return Rain100HDataset(root_dir=as_str(DATA["rain100h"] / "test"), transform=tfm)
    if name == "Rain100L":
        return Rain100LDataset(root_dir=as_str(DATA["rain100l"] / "test"), transform=tfm)
    if name == "GoPro":
        return GoProDataset(as_str(DATA["gopro"] / "gopro_test_pairs.csv"), transform=tfm)
    if name == "RESIDE":
        return RESIDEDataset(root_dir=as_str(DATA["reside"]), split="test", transform=tfm, strict=True)
    if name == "CSD":
        return CSDDataset(root_dir=as_str(DATA["csd"]), split="test", transform=tfm)
    raise ValueError(name)


def write_eval_options(path, ckpt, daclip_weight, name):
    text = f"""name: {name}
suffix: ~
model: denoising
distortion: [rain100h,rain100l,motion-blurry,noisy,snowy]
gpu_ids: [0]

sde:
  max_sigma: 50
  T: 100
  schedule: cosine
  eps: 0.005
  sampling_mode: posterior

degradation:
  sigma: 25
  noise_type: G
  scale: 4

datasets:
  test1:
   name: Test
   mode: LQGT
   dataroot_GT: unused
   dataroot_LQ: unused

network_G:
  which_model_G: ConditionalUNet
  setting:
    in_nc: 3
    out_nc: 3
    nf: 64
    ch_mult: [1, 2, 4, 8]
    context_dim: 512
    use_degra_context: true
    use_image_context: true

path:
  pretrain_model_G: {str(ckpt).replace("\\", "/")}
  daclip: {str(daclip_weight).replace("\\", "/")}
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


@torch.no_grad()
def eval_one_dataset(model, clip_model, sde, dataset_name, max_samples=None):
    ds = ClipWrappedDataset(build_base_dataset(dataset_name))
    if max_samples is not None:
        ds = Subset(ds, list(range(min(max_samples, len(ds)))))
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, pin_memory=(DEVICE.type == "cuda"))

    psnrs, ssims = [], []
    for batch in tqdm(loader, desc=f"Testing {dataset_name}"):
        lq = batch["LQ"].to(DEVICE)
        gt = batch["GT"].to(DEVICE)
        lq_clip = batch["LQ_clip"].to(DEVICE)

        with torch.cuda.amp.autocast(enabled=(DEVICE.type == "cuda")):
            image_context, degra_context = clip_model.encode_image(lq_clip, control=True)
            image_context = image_context.float()
            degra_context = degra_context.float()

        noisy_state = sde.noise_state(lq)
        model.feed_data(noisy_state, lq, gt, text_context=degra_context, image_context=image_context)
        model.test(sde, mode="posterior", save_states=False)
        visuals = model.get_current_visuals()
        pred = visuals["Output"].unsqueeze(0).to(DEVICE).clamp(0, 1)
        psnr, ssim = compute_metrics(pred, gt)
        psnrs.append(psnr)
        ssims.append(ssim)

    return float(np.mean(psnrs)), float(np.mean(ssims)), len(psnrs)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a P2 all-in-one DA-CLIP baseline.")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--daclip-weight", default=str(DACLP_ROOT / "pretrained" / "daclip_ViT-B-32.pt"))
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Optional cap per dataset. Full DA-CLIP SDE evaluation can be very slow.")
    parser.add_argument("--row-status", default="Retrained")
    parser.add_argument("--training-data", default="Same as VTN")
    parser.add_argument("--result-name", default="p2_daclip_baseline")
    return parser.parse_args()


def main():
    args = parse_args()
    ckpt = Path(args.ckpt)
    daclip_weight = Path(args.daclip_weight)
    if not ckpt.exists():
        raise FileNotFoundError(ckpt)
    if not daclip_weight.exists():
        raise FileNotFoundError(daclip_weight)

    opt_path = ROOT_DIR / "experiments" / "daclip_eval_tmp.yml"
    write_eval_options(opt_path, ckpt, daclip_weight, args.result_name)
    opt = option.dict_to_nonedict(option.parse(str(opt_path), is_train=False))

    model = create_model(opt)
    sde = daclip_utils.IRSDE(
        max_sigma=opt["sde"]["max_sigma"],
        T=opt["sde"]["T"],
        schedule=opt["sde"]["schedule"],
        eps=opt["sde"]["eps"],
        device=DEVICE,
    )
    sde.set_model(model.model)

    clip_model, _ = open_clip.create_model_from_pretrained("daclip_ViT-B-32", pretrained=str(daclip_weight))
    clip_model = clip_model.to(DEVICE).eval()

    names = ["Rain100H", "Rain100L", "GoPro", "RESIDE", "CSD"]
    results = {}
    for name in names:
        psnr, ssim, n = eval_one_dataset(model, clip_model, sde, name, args.max_samples)
        results[name] = (psnr, ssim, n)
        print(f"[{name}] N={n} PSNR={psnr:.2f} SSIM={ssim:.4f}")

    rain_psnr = (results["Rain100H"][0] + results["Rain100L"][0]) / 2
    rain_ssim = (results["Rain100H"][1] + results["Rain100L"][1]) / 2
    avg_psnr = (rain_psnr + results["GoPro"][0] + results["RESIDE"][0] + results["CSD"][0]) / 4
    avg_ssim = (rain_ssim + results["GoPro"][1] + results["RESIDE"][1] + results["CSD"][1]) / 4

    suffix = f" (N={args.max_samples})" if args.max_samples is not None else ""
    row = {
        "Method": "DA-CLIP",
        "Status": args.row_status,
        "Direct Comparison?": "Yes",
        "Training Data": args.training_data + suffix,
        "Rain(avg)": f"{rain_psnr:.2f} / {rain_ssim:.4f}",
        "GoPro": f"{results['GoPro'][0]:.2f} / {results['GoPro'][1]:.4f}",
        "RESIDE-6K": f"{results['RESIDE'][0]:.2f} / {results['RESIDE'][1]:.4f}",
        "CSD": f"{results['CSD'][0]:.2f} / {results['CSD'][1]:.4f}",
        "Avg": f"{avg_psnr:.2f} / {avg_ssim:.4f}",
        "Checkpoint": str(ckpt),
        "Protocol": f"P2 all-in-one DA-CLIP; max_samples={args.max_samples}",
    }
    out = ROOT_DIR / "experiments" / "table1_daclip_row.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)

    md_out = out.with_suffix(".md")
    with md_out.open("w", encoding="utf-8") as f:
        f.write("| Method | Status | Direct Comparison? | Training Data | Rain(avg) | GoPro | RESIDE-6K | CSD | Avg |\n")
        f.write("|---|---|---|---|---:|---:|---:|---:|---:|\n")
        f.write(
            f"| DA-CLIP | {row['Status']} | Yes | {row['Training Data']} | {row['Rain(avg)']} | "
            f"{row['GoPro']} | {row['RESIDE-6K']} | {row['CSD']} | {row['Avg']} |\n"
        )
    print(f"[Table 1 CSV] {out}")
    print(f"[Table 1 row] {md_out}")


if __name__ == "__main__":
    main()
