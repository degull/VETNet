# train_voc2012_head_only.py
# ------------------------------------------------------------
# VOC2012 detection head-only training using a frozen Restormer+Volterra encoder.
# - Your VOC layout is supported.
# - Backbone weights loaded from your restorer ckpt (RestormerVolterra).
# - ✅ Auto-infer dim / bias / volterra_rank from checkpoint keys.
# - ✅ Debug-print ckpt key prefixes & samples.
# - ✅ Sanity batch print.
# - ✅ First-iter loss print.
# - ✅ Optional qualitative visualization each epoch (GT vs Pred).
# - ✅ Terminal progress with ETA + current performance (loss + mAP if available)
#
# ✅ NEW (requested):
# (1) FasterRCNN min/max_size options (reduces resize cost -> faster)
# (2) AMP option (autocast + GradScaler)
# (3) Eval sampling option (evaluate only first N images for speed)
# (4) DataLoader persistent_workers option
#
# Usage (PowerShell):
#   cd "E:\restormer+volterra\Restormer + Volterra"
#   python -u .\train_voc2012_head_only.py `
#     --voc_root "E:\restormer+volterra\data\VOC" `
#     --restorer_ckpt "E:\restormer+volterra\checkpoints\restormer_volterra_gopro\epoch_012_ssim0.9692_psnr34.75.pth" `
#     --epochs 12 `
#     --batch_size 2 `
#     --num_workers 2 `
#     --persistent_workers 1 `
#     --use_amp 1 `
#     --print_every 200 `
#     --eval_every 1 `
#     --eval_max_images 2000 `
#     --det_min_size 512 `
#     --det_max_size 512 `
#     --viz_every 1 `
#     --viz_num 8 `
#     --viz_thr 0.5 `
#     --viz_dir ".\voc2012_head_only_runs\viz"
# ------------------------------------------------------------

import os
import time
import random
import argparse
from collections import OrderedDict, Counter
import math

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from PIL import Image, ImageDraw, ImageFont
import xml.etree.ElementTree as ET

from torchvision.transforms import functional as TF
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign


# -----------------------------
# 0) Pascal VOC class mapping
# -----------------------------
VOC_CLASSES = [
    "aeroplane","bicycle","bird","boat","bottle","bus","car","cat","chair","cow",
    "diningtable","dog","horse","motorbike","person","pottedplant","sheep","sofa","train","tvmonitor"
]
CLASS_TO_IDX = {c: i+1 for i, c in enumerate(VOC_CLASSES)}  # 0 is background
IDX_TO_CLASS = {i+1: c for i, c in enumerate(VOC_CLASSES)}


# -----------------------------
# Utils: time formatting
# -----------------------------
def _fmt_time(sec: float) -> str:
    if sec is None or not math.isfinite(sec) or sec < 0:
        return "N/A"
    sec = int(sec)
    h = sec // 3600
    m = (sec % 3600) // 60
    s = sec % 60
    if h > 0:
        return f"{h:d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


# -----------------------------
# 1) VOC Dataset (custom 2012 layout)
# -----------------------------
class VOC2012CustomDataset(Dataset):
    """
    Robust VOC loader for your custom layout.
    If ImageSets/Main/{split}.txt is missing, it will auto-build ids from Annotations/*.xml.

    Your structure:
      VOC_ROOT/
        VOC2012_train_val/VOC2012_train_val/{JPEGImages,Annotations,ImageSets}
        VOC2012_test/VOC2012_test/{JPEGImages,Annotations,ImageSets}
    """
    def __init__(self, voc_root, subset="VOC2012_train_val", split_name="trainval",
                 transforms=None, keep_difficult=False):
        super().__init__()
        self.voc_root = voc_root
        self.subset = subset
        self.split_name = split_name
        self.transforms = transforms
        self.keep_difficult = keep_difficult

        base = os.path.join(voc_root, subset, subset)
        self.img_dir = os.path.join(base, "JPEGImages")
        self.ann_dir = os.path.join(base, "Annotations")

        main_dir = os.path.join(base, "ImageSets", "Main")
        layout_dir = os.path.join(base, "ImageSets", "Layout")

        cand_files = [
            os.path.join(main_dir, f"{split_name}.txt"),
            os.path.join(layout_dir, f"{split_name}.txt"),
        ]
        split_file = None
        for f in cand_files:
            if os.path.exists(f):
                split_file = f
                break

        if split_file is not None:
            with open(split_file, "r", encoding="utf-8") as f:
                ids = []
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    ids.append(line.split()[0])
            self.ids = ids
            print(f"[VOC] subset={subset} split={split_name} | ids={len(self.ids)} | split_file={split_file}", flush=True)
        else:
            if not os.path.isdir(self.ann_dir):
                raise FileNotFoundError(f"Annotations dir not found: {self.ann_dir}")
            xmls = [p for p in os.listdir(self.ann_dir) if p.lower().endswith(".xml")]
            self.ids = [os.path.splitext(p)[0] for p in sorted(xmls)]
            print(f"[VOC] subset={subset} split={split_name} | ids={len(self.ids)} | split_file=None (auto from Annotations)", flush=True)

    def __len__(self):
        return len(self.ids)

    def _parse_xml(self, xml_path):
        tree = ET.parse(xml_path)
        root = tree.getroot()

        boxes, labels, iscrowd = [], [], []
        for obj in root.findall("object"):
            name = obj.find("name").text.strip()
            if name not in CLASS_TO_IDX:
                continue

            difficult = int(obj.find("difficult").text) if obj.find("difficult") is not None else 0
            if (not self.keep_difficult) and difficult == 1:
                continue

            bnd = obj.find("bndbox")
            xmin = float(bnd.find("xmin").text)
            ymin = float(bnd.find("ymin").text)
            xmax = float(bnd.find("xmax").text)
            ymax = float(bnd.find("ymax").text)

            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(CLASS_TO_IDX[name])
            iscrowd.append(0)

        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            iscrowd = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)
            iscrowd = torch.tensor(iscrowd, dtype=torch.int64)

        return boxes, labels, iscrowd

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img_path_jpg = os.path.join(self.img_dir, f"{img_id}.jpg")
        img_path_png = os.path.join(self.img_dir, f"{img_id}.png")
        xml_path = os.path.join(self.ann_dir, f"{img_id}.xml")

        if os.path.exists(img_path_jpg):
            img_path = img_path_jpg
        elif os.path.exists(img_path_png):
            img_path = img_path_png
        else:
            raise FileNotFoundError(f"Image not found for id={img_id}: {img_path_jpg} / {img_path_png}")

        img = Image.open(img_path).convert("RGB")
        boxes, labels, iscrowd = self._parse_xml(xml_path)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx], dtype=torch.int64),
            "iscrowd": iscrowd,
            # string metadata for viz naming (train loop will not .to() this)
            "img_id_str": img_id,
        }

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target


# -----------------------------
# 2) Transforms
# -----------------------------
class Compose:
    def __init__(self, ts):
        self.ts = ts
    def __call__(self, img, target):
        for t in self.ts:
            img, target = t(img, target)
        return img, target

class ToTensor:
    def __call__(self, img, target):
        return TF.to_tensor(img), target

class RandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p
    def __call__(self, img, target):
        if random.random() < self.p:
            img = TF.hflip(img)
            if "boxes" in target:
                w = img.shape[-1]
                boxes = target["boxes"].clone()
                boxes[:, 0] = w - target["boxes"][:, 2]
                boxes[:, 2] = w - target["boxes"][:, 0]
                target["boxes"] = boxes
        return img, target

def get_transforms(train=True):
    if train:
        return Compose([ToTensor(), RandomHorizontalFlip(0.5)])
    else:
        return Compose([ToTensor()])

def collate_fn(batch):
    return tuple(zip(*batch))


# -----------------------------
# 3) Frozen backbone wrapper
# -----------------------------
class FrozenBackbone(nn.Module):
    """
    Must output a single feature map [B,C,H,W].
    """
    def __init__(self, encoder: nn.Module, out_channels: int):
        super().__init__()
        self.encoder = encoder
        self.out_channels = out_channels
        for p in self.encoder.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def forward(self, x):
        return self.encoder(x)


# -----------------------------
# 3.1) CKPT helpers (infer dim/bias/rank + debug keys)
# -----------------------------
def infer_state_dict_from_ckpt(ckpt_obj):
    """
    Try common nesting patterns and return the best-guess state_dict.
    """
    if isinstance(ckpt_obj, dict):
        for key in ["state_dict", "model", "net", "params", "model_state", "state"]:
            if key in ckpt_obj and isinstance(ckpt_obj[key], dict):
                return ckpt_obj[key]
        if any(isinstance(v, torch.Tensor) for v in ckpt_obj.values()):
            return ckpt_obj
    raise ValueError("Cannot infer state_dict from ckpt. Inspect checkpoint structure.")

def summarize_prefixes(keys, topk=10):
    def prefix_of(k: str):
        if "." in k:
            return k.split(".", 1)[0] + "."
        return "(no_dot)"
    cnt = Counter(prefix_of(k) for k in keys)
    return cnt.most_common(topk)

def strip_prefix_if_present(sd, prefixes=("module.", "model.", "net.")):
    out = {}
    stripped = 0
    for k, v in sd.items():
        nk = k
        for p in prefixes:
            if nk.startswith(p):
                nk = nk[len(p):]
                stripped += 1
        out[nk] = v
    return out, stripped

def infer_volterra_rank_from_state(state: dict) -> int:
    """
    Detect max index in keys like:
      encoder1.body.0.volterra1.W2a.3.weight -> rank = 4
    """
    max_idx = -1
    for k in state.keys():
        if ".volterra" in k and ".W2a." in k:
            tail = k.split(".W2a.", 1)[1]   # "3.weight" ...
            idx_str = tail.split(".", 1)[0]
            if idx_str.isdigit():
                max_idx = max(max_idx, int(idx_str))
    return max_idx + 1 if max_idx >= 0 else 2

def infer_bias_from_state(state: dict) -> bool:
    """
    If common conv bias keys exist, bias=True; else bias=False.
    """
    probe_bias_keys = [
        "encoder1.body.0.attn.qkv.bias",
        "encoder1.body.0.attn.dwconv.bias",
        "encoder1.body.0.attn.project_out.bias",
        "encoder1.body.0.ffn.project_in.bias",
        "encoder1.body.0.ffn.dwconv.bias",
        "encoder1.body.0.ffn.project_out.bias",
    ]
    return any(k in state for k in probe_bias_keys)


def build_frozen_backbone_from_restorer(
    ckpt_path: str,
    device: torch.device,
    debug_keys: int = 20,
    auto_strip_prefix: int = 1,
    dummy_hw: int = 256,
):
    """
    ✅ Auto-infer dim / bias / volterra_rank from checkpoint
    ✅ Print prefix stats + a few sample keys
    ✅ Report matched key ratio (shape-matched)
    """
    import sys
    sys.path.append(r"E:/restormer+volterra/Restormer + Volterra")

    # IMPORTANT: this import must match your project structure.
    from models.restormer_volterra import RestormerVolterra

    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = infer_state_dict_from_ckpt(ckpt)

    keys = list(state.keys())
    print(f"[CKPT] raw state keys: {len(keys)}", flush=True)
    print(f"[CKPT] raw prefix top: {summarize_prefixes(keys, topk=8)}", flush=True)
    print("[CKPT] raw sample keys:", flush=True)
    for k in keys[:max(1, debug_keys)]:
        print(f"  - {k}", flush=True)

    if auto_strip_prefix:
        state2, stripped = strip_prefix_if_present(state, prefixes=("module.", "model.", "net."))
        if stripped > 0:
            state = state2
            keys2 = list(state.keys())
            print(f"[CKPT] stripped common prefixes (hits={stripped})", flush=True)
            print(f"[CKPT] after-strip prefix top: {summarize_prefixes(keys2, topk=8)}", flush=True)

    if "patch_embed.weight" not in state:
        cand = [k for k in state.keys() if k.endswith("patch_embed.weight")]
        raise KeyError(f"Missing 'patch_embed.weight'. Candidates: {cand[:10]}")
    dim = int(state["patch_embed.weight"].shape[0])

    bias = infer_bias_from_state(state)
    volterra_rank = infer_volterra_rank_from_state(state)

    print(f"[CKPT] inferred dim={dim}, bias={int(bias)}, volterra_rank={volterra_rank}", flush=True)

    model = RestormerVolterra(dim=dim, bias=bias, volterra_rank=volterra_rank)

    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[CKPT] load_state_dict strict=False | missing={len(missing)} unexpected={len(unexpected)}", flush=True)

    model_sd = model.state_dict()
    matched = 0
    total = 0
    for k, v in model_sd.items():
        total += 1
        if k in state and tuple(state[k].shape) == tuple(v.shape):
            matched += 1
    print(f"[CKPT] matched_keys={matched}/{total} ({100.0*matched/max(1,total):.2f}%)", flush=True)

    if len(unexpected) > 0:
        print("[CKPT] unexpected sample:", flush=True)
        for k in unexpected[:20]:
            print(f"  - {k}", flush=True)
    if len(missing) > 0:
        print("[CKPT] missing sample:", flush=True)
        for k in missing[:20]:
            print(f"  - {k}", flush=True)

    model.to(device).eval()

    # Your RestormerVolterra has encoder1/2/3/latent not a single `encoder`.
    # For head-only detection, we use a wrapper to output a single feature map.
    class EncoderWrapper(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m
        def forward(self, x):
            x = self.m.patch_embed(x)
            x = self.m.encoder1(x)
            return x

    encoder = EncoderWrapper(model)

    with torch.no_grad():
        dummy = torch.zeros(1, 3, dummy_hw, dummy_hw).to(device)
        feat = encoder(dummy)
        if isinstance(feat, (list, tuple)):
            feat = feat[-1]
        out_channels = int(feat.shape[1])

    backbone = FrozenBackbone(encoder, out_channels=out_channels).to(device)
    backbone.eval()
    print(f"[Backbone] frozen encoder out_channels={out_channels}", flush=True)
    return backbone


# -----------------------------
# 4) Build Faster R-CNN with custom backbone
# -----------------------------
def build_faster_rcnn_with_custom_backbone(
    backbone: nn.Module,
    num_classes: int = 21,
    det_min_size: int = 512,
    det_max_size: int = 512,
):
    anchor_generator = AnchorGenerator(
        sizes=((32, 64, 128, 256, 512),),
        aspect_ratios=((0.5, 1.0, 2.0),)
    )

    roi_pooler = MultiScaleRoIAlign(
        featmap_names=["0"],
        output_size=7,
        sampling_ratio=2
    )

    class BackboneAdapter(nn.Module):
        def __init__(self, bb):
            super().__init__()
            self.bb = bb
            self.out_channels = bb.out_channels
        def forward(self, x):
            feat = self.bb(x)
            return OrderedDict([("0", feat)])

    # ✅ min_size / max_size control internal resize. Big speed lever.
    model = FasterRCNN(
        backbone=BackboneAdapter(backbone),
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler,
        min_size=det_min_size,
        max_size=det_max_size,
    )
    return model


# -----------------------------
# 5) mAP@0.5 Evaluation (torchmetrics) + optional sampling
# -----------------------------
@torch.no_grad()
def evaluate_map50(model, loader, device, max_images: int = -1):
    """
    max_images:
      -1 => evaluate full loader
       N => evaluate only first N images for speed
    """
    try:
        from torchmetrics.detection.mean_ap import MeanAveragePrecision
    except Exception:
        print("[Eval] torchmetrics not available -> skip mAP. Install: pip install torchmetrics", flush=True)
        return None

    metric = MeanAveragePrecision(iou_type="bbox", iou_thresholds=[0.5])
    model.eval()

    seen = 0
    for images, targets in loader:
        images = [img.to(device) for img in images]
        outputs = model(images)

        preds = []
        gts = []
        for out, tgt in zip(outputs, targets):
            preds.append({
                "boxes": out["boxes"].detach().cpu(),
                "scores": out["scores"].detach().cpu(),
                "labels": out["labels"].detach().cpu(),
            })
            gts.append({
                "boxes": tgt["boxes"].detach().cpu(),
                "labels": tgt["labels"].detach().cpu(),
            })
            seen += 1
            if max_images > 0 and seen >= max_images:
                break

        metric.update(preds, gts)

        if max_images > 0 and seen >= max_images:
            break

    res = metric.compute()
    return float(res["map"])


# -----------------------------
# 6) Visualization during training
# -----------------------------
def _load_font(font_size=16):
    try:
        return ImageFont.truetype("arial.ttf", font_size)
    except Exception:
        return ImageFont.load_default()

def _draw_boxes(pil_img, boxes, labels=None, scores=None, color=(255, 0, 0), width=3):
    img = pil_img.copy()
    draw = ImageDraw.Draw(img)
    font = _load_font(16)

    for i, b in enumerate(boxes):
        x1, y1, x2, y2 = [float(v) for v in b]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=width)

        text_parts = []
        if labels is not None:
            cls = IDX_TO_CLASS.get(int(labels[i]), str(int(labels[i])))
            text_parts.append(cls)
        if scores is not None:
            text_parts.append(f"{float(scores[i]):.2f}")

        if text_parts:
            text = " ".join(text_parts)
            tb = draw.textbbox((0, 0), text, font=font)
            tw, th = tb[2] - tb[0], tb[3] - tb[1]
            draw.rectangle([x1, y1 - th - 4, x1 + tw + 6, y1], fill=(0, 0, 0))
            draw.text((x1 + 3, y1 - th - 2), text, fill=(255, 255, 255), font=font)

    return img

def _concat_side_by_side(img_left, img_right):
    w1, h1 = img_left.size
    w2, h2 = img_right.size
    H = max(h1, h2)
    canvas = Image.new("RGB", (w1 + w2, H), (0, 0, 0))
    canvas.paste(img_left, (0, 0))
    canvas.paste(img_right, (w1, 0))
    return canvas

@torch.no_grad()
def visualize_predictions(model, dataset, device, out_dir, epoch, indices, score_thr=0.5, max_det=50, draw_gt=True):
    os.makedirs(out_dir, exist_ok=True)
    model.eval()

    base = os.path.join(dataset.voc_root, dataset.subset, dataset.subset)
    img_dir = os.path.join(base, "JPEGImages")
    ann_dir = os.path.join(base, "Annotations")

    for k, idx in enumerate(indices, 1):
        img_id = dataset.ids[idx]

        img_path_jpg = os.path.join(img_dir, f"{img_id}.jpg")
        img_path_png = os.path.join(img_dir, f"{img_id}.png")
        xml_path = os.path.join(ann_dir, f"{img_id}.xml")

        img_path = img_path_jpg if os.path.exists(img_path_jpg) else img_path_png
        pil = Image.open(img_path).convert("RGB")
        x = TF.to_tensor(pil).to(device)

        out = model([x])[0]
        boxes = out["boxes"].detach().cpu()
        scores = out["scores"].detach().cpu()
        labels = out["labels"].detach().cpu()

        keep = scores >= score_thr
        boxes = boxes[keep][:max_det]
        scores = scores[keep][:max_det]
        labels = labels[keep][:max_det]

        pred_img = _draw_boxes(pil, boxes, labels=labels, scores=scores, color=(255, 0, 0), width=3)

        if draw_gt:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            gt_boxes = []
            gt_labels = []
            for obj in root.findall("object"):
                name = obj.find("name").text.strip()
                if name not in CLASS_TO_IDX:
                    continue
                bnd = obj.find("bndbox")
                xmin = float(bnd.find("xmin").text)
                ymin = float(bnd.find("ymin").text)
                xmax = float(bnd.find("xmax").text)
                ymax = float(bnd.find("ymax").text)
                gt_boxes.append([xmin, ymin, xmax, ymax])
                gt_labels.append(CLASS_TO_IDX[name])

            if len(gt_boxes) == 0:
                gt_boxes = torch.zeros((0, 4), dtype=torch.float32)
                gt_labels = torch.zeros((0,), dtype=torch.int64)
            else:
                gt_boxes = torch.tensor(gt_boxes, dtype=torch.float32)
                gt_labels = torch.tensor(gt_labels, dtype=torch.int64)

            gt_img = _draw_boxes(pil, gt_boxes, labels=gt_labels, scores=None, color=(0, 255, 0), width=3)
            merged = _concat_side_by_side(gt_img, pred_img)
            save_path = os.path.join(out_dir, f"epoch_{epoch:03d}_{k:02d}_{img_id}_GT_PRED_thr{score_thr:.2f}.png")
            merged.save(save_path)
        else:
            save_path = os.path.join(out_dir, f"epoch_{epoch:03d}_{k:02d}_{img_id}_PRED_thr{score_thr:.2f}.png")
            pred_img.save(save_path)

        print(f"[Viz] saved {save_path}", flush=True)


# -----------------------------
# 7) Train loop with ETA + current performance + AMP
# -----------------------------
def train_one_epoch(
    model,
    optimizer,
    loader,
    device,
    epoch,
    print_every=200,
    epoch_start_global_step=0,
    total_global_steps=None,
    last_map50=None,
    best_map50=None,
    global_start_time=None,
    use_amp: bool = False,
    scaler=None,
):
    """
    Prints:
      - avg loss so far
      - it/s
      - ETA for epoch
      - ETA for total training
      - last/best mAP (if available)
    """
    model.train()
    epoch_start_time = time.time()

    total_loss = 0.0
    n = 0

    last_print_time = time.time()
    last_print_it = 0

    iters_per_epoch = len(loader)

    for it, (images, targets) in enumerate(loader, 1):
        images = [img.to(device) for img in images]

        # ✅ move only tensors to device (keep strings like img_id_str)
        targets = [
            {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in t.items()}
            for t in targets
        ]

        optimizer.zero_grad(set_to_none=True)

        # ✅ AMP forward
        if use_amp:
            with torch.cuda.amp.autocast(enabled=True):
                loss_dict = model(images, targets)
                loss = sum(loss_dict.values())
            if it == 1:
                print(f"[FirstIter] loss_dict keys={list(loss_dict.keys())} loss={float(loss.item()):.4f}", flush=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())
            if it == 1:
                print(f"[FirstIter] loss_dict keys={list(loss_dict.keys())} loss={float(loss.item()):.4f}", flush=True)
            loss.backward()
            optimizer.step()

        total_loss += float(loss.item())
        n += 1

        if (it % print_every == 0) or (it == iters_per_epoch):
            now = time.time()
            avg_loss = total_loss / max(1, n)

            # iter/sec (since last print)
            dt = now - last_print_time
            dit = it - last_print_it
            it_per_sec = (dit / dt) if dt > 0 else 0.0

            # epoch ETA
            remain_it_epoch = iters_per_epoch - it
            eta_epoch = (remain_it_epoch / it_per_sec) if it_per_sec > 0 else None

            # total ETA (using global average speed)
            if global_start_time is not None and total_global_steps is not None:
                global_elapsed = now - global_start_time
                done_steps = epoch_start_global_step + it
                steps_left = max(0, total_global_steps - done_steps)
                global_steps_per_sec = (done_steps / global_elapsed) if global_elapsed > 0 else 0.0
                eta_total = (steps_left / global_steps_per_sec) if global_steps_per_sec > 0 else None
            else:
                eta_total = None

            mtxt = ""
            if last_map50 is not None:
                mtxt += f" | last mAP@0.5={last_map50:.4f}"
            if best_map50 is not None and best_map50 >= 0:
                mtxt += f" | best mAP@0.5={best_map50:.4f}"

            print(
                f"[Epoch {epoch:03d} | Iter {it:05d}/{iters_per_epoch:05d}] "
                f"loss(avg)={avg_loss:.4f} | it/s={it_per_sec:.2f} "
                f"| ETA(epoch)={_fmt_time(eta_epoch)} | ETA(total)={_fmt_time(eta_total)}{mtxt}",
                flush=True
            )

            last_print_time = now
            last_print_it = it

    epoch_time = time.time() - epoch_start_time
    return total_loss / max(1, n), epoch_time


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--voc_root", type=str, default=r"E:\restormer+volterra\data\VOC")
    parser.add_argument("--restorer_ckpt", type=str, required=True)

    parser.add_argument("--train_subset", type=str, default="VOC2012_train_val")
    parser.add_argument("--test_subset", type=str, default="VOC2012_test")

    parser.add_argument("--train_split", type=str, default="trainval")
    parser.add_argument("--test_split", type=str, default="test")

    parser.add_argument("--out_dir", type=str, default="./voc2012_head_only_runs")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch_size", type=int, default=2)

    # dataloader
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--persistent_workers", type=int, default=0, help="1 to keep workers alive (num_workers>0)")
    parser.add_argument("--prefetch_factor", type=int, default=2, help="only if num_workers>0")
    parser.add_argument("--pin_memory", type=int, default=1)

    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval_every", type=int, default=1)
    parser.add_argument("--print_every", type=int, default=200)

    # ✅ detector resize control
    parser.add_argument("--det_min_size", type=int, default=512)
    parser.add_argument("--det_max_size", type=int, default=512)

    # ✅ AMP
    parser.add_argument("--use_amp", type=int, default=0)

    # ✅ eval sampling
    parser.add_argument("--eval_max_images", type=int, default=-1, help="-1 full eval, else first N images")

    # viz
    parser.add_argument("--viz_every", type=int, default=1, help="visualize every N epochs (0 to disable)")
    parser.add_argument("--viz_num", type=int, default=8, help="number of images to visualize")
    parser.add_argument("--viz_thr", type=float, default=0.5, help="score threshold for visualization")
    parser.add_argument("--viz_dir", type=str, default="./voc2012_head_only_runs/viz")
    parser.add_argument("--viz_max_det", type=int, default=50)
    parser.add_argument("--viz_draw_gt", type=int, default=1, help="1: save GT vs Pred side-by-side")

    # ckpt debug
    parser.add_argument("--ckpt_debug_keys", type=int, default=20, help="print first N keys from ckpt state_dict")
    parser.add_argument("--ckpt_auto_strip_prefix", type=int, default=1, help="auto strip module./model./net. prefixes")

    # dummy input size for channel inference
    parser.add_argument("--dummy_hw", type=int, default=256)

    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.viz_dir, exist_ok=True)

    # datasets
    train_ds = VOC2012CustomDataset(
        args.voc_root, subset=args.train_subset, split_name=args.train_split,
        transforms=get_transforms(train=True)
    )
    test_ds = VOC2012CustomDataset(
        args.voc_root, subset=args.test_subset, split_name=args.test_split,
        transforms=get_transforms(train=False)
    )

    # dataloader flags
    use_persistent = bool(args.persistent_workers) and args.num_workers > 0
    dl_kwargs = dict(
        num_workers=args.num_workers,
        pin_memory=bool(args.pin_memory),
        collate_fn=collate_fn,
    )
    if args.num_workers > 0:
        dl_kwargs["persistent_workers"] = use_persistent
        dl_kwargs["prefetch_factor"] = args.prefetch_factor

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, **dl_kwargs
    )
    test_loader = DataLoader(
        test_ds, batch_size=1, shuffle=False, **dl_kwargs
    )

    # sanity
    print("[Sanity] loading one batch...", flush=True)
    images, targets = next(iter(train_loader))
    print(f"[Sanity] batch loaded: n={len(images)} img0={tuple(images[0].shape)} boxes0={targets[0]['boxes'].shape}", flush=True)

    # fixed viz indices
    viz_indices = list(range(len(test_ds)))
    random.shuffle(viz_indices)
    viz_indices = viz_indices[:args.viz_num]
    print(f"[Viz] fixed indices for qualitative: {viz_indices}", flush=True)

    # backbone
    backbone = build_frozen_backbone_from_restorer(
        args.restorer_ckpt,
        device=device,
        debug_keys=args.ckpt_debug_keys,
        auto_strip_prefix=args.ckpt_auto_strip_prefix,
        dummy_hw=args.dummy_hw,
    )

    # detector (✅ min/max size applied)
    model = build_faster_rcnn_with_custom_backbone(
        backbone,
        num_classes=21,
        det_min_size=args.det_min_size,
        det_max_size=args.det_max_size,
    ).to(device)

    # optimizer (head-only)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in trainable_params)
    print(f"[Model] total params: {total/1e6:.2f}M | trainable(head-only): {trainable/1e6:.2f}M", flush=True)

    optimizer = torch.optim.SGD(trainable_params, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    lr_sched = torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1, args.epochs // 3), gamma=0.1)

    # ✅ AMP scaler
    use_amp = bool(args.use_amp) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_map = -1.0
    last_map = None

    # global ETA setup
    global_start_time = time.time()
    iters_per_epoch = len(train_loader)
    total_global_steps = args.epochs * iters_per_epoch
    print(f"[Prog] iters/epoch={iters_per_epoch} | total_steps={total_global_steps}", flush=True)
    print(f"[Cfg] det_min_size={args.det_min_size} det_max_size={args.det_max_size} | AMP={int(use_amp)} | "
          f"eval_max_images={args.eval_max_images} | persistent_workers={int(use_persistent)}",
          flush=True)

    for epoch in range(1, args.epochs + 1):
        epoch_start_step = (epoch - 1) * iters_per_epoch

        loss, epoch_time = train_one_epoch(
            model=model,
            optimizer=optimizer,
            loader=train_loader,
            device=device,
            epoch=epoch,
            print_every=args.print_every,
            epoch_start_global_step=epoch_start_step,
            total_global_steps=total_global_steps,
            last_map50=last_map,
            best_map50=best_map if best_map >= 0 else None,
            global_start_time=global_start_time,
            use_amp=use_amp,
            scaler=scaler,
        )
        lr_sched.step()

        elapsed_total = time.time() - global_start_time
        avg_epoch_time = elapsed_total / max(1, epoch)
        remain_epochs = args.epochs - epoch
        eta_total_epochs = remain_epochs * avg_epoch_time

        print(
            f"[Epoch {epoch:03d}] train_loss={loss:.4f} | epoch_time={_fmt_time(epoch_time)} "
            f"| ETA(total,epoch-avg)={_fmt_time(eta_total_epochs)}",
            flush=True
        )

        # eval (✅ sampling)
        if epoch % args.eval_every == 0:
            eval_t0 = time.time()
            map50 = evaluate_map50(model, test_loader, device, max_images=args.eval_max_images)
            eval_dt = time.time() - eval_t0

            if map50 is not None:
                last_map = map50
                eval_tag = f"(first {args.eval_max_images})" if args.eval_max_images and args.eval_max_images > 0 else "(full)"
                print(f"[Epoch {epoch:03d}] VOC test mAP@0.5 {eval_tag} = {map50:.4f} | eval_time={_fmt_time(eval_dt)}", flush=True)

                if map50 > best_map:
                    best_map = map50
                    save_path = os.path.join(args.out_dir, f"best_map50_{best_map:.4f}.pth")
                    torch.save({"model": model.state_dict(), "epoch": epoch, "map50": best_map}, save_path)
                    print(f"[Save] {save_path}", flush=True)
            else:
                save_path = os.path.join(args.out_dir, f"epoch_{epoch:03d}.pth")
                torch.save({"model": model.state_dict(), "epoch": epoch}, save_path)
                print(f"[Save] {save_path}", flush=True)

        # viz
        if args.viz_every > 0 and (epoch % args.viz_every == 0):
            visualize_predictions(
                model=model,
                dataset=test_ds,
                device=device,
                out_dir=args.viz_dir,
                epoch=epoch,
                indices=viz_indices,
                score_thr=args.viz_thr,
                max_det=args.viz_max_det,
                draw_gt=bool(args.viz_draw_gt),
            )

    print(f"[Done] best mAP@0.5 = {best_map:.4f}", flush=True)


if __name__ == "__main__":
    main()