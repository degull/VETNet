import os
import random
from typing import List, Tuple

import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
from torch.utils.data import Dataset
import torchvision.transforms as T

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def list_images(root: str) -> List[str]:
    if not os.path.isdir(root):
        raise FileNotFoundError(f"clean_root not found: {root}")
    files = []
    for fn in os.listdir(root):
        if fn.lower().endswith(IMG_EXTS):
            files.append(os.path.join(root, fn))
    files.sort()
    return files


# ---------------------------
# Distortion ops (PIL-based)
# ---------------------------
def apply_gaussian_blur(img: Image.Image, rng: random.Random) -> Image.Image:
    r = rng.uniform(0.2, 2.0)
    return img.filter(ImageFilter.GaussianBlur(radius=r))


def apply_motion_blur(img: Image.Image, rng: random.Random) -> Image.Image:
    """
    Safe motion blur (NO ImageFilter.Kernel).
    Shift-average along a direction, always stable across Pillow versions.
    """
    arr = np.array(img).astype(np.float32) / 255.0  # HWC, [0,1]

    # blur length (number of shifts)
    L = int(rng.randint(3, 21))  # >=3
    angle = rng.choice([0, 45, 90, 135])

    # direction (dx, dy)
    if angle == 0:        # horizontal
        dx, dy = 1, 0
    elif angle == 90:     # vertical
        dx, dy = 0, 1
    elif angle == 45:     # diag down-right
        dx, dy = 1, 1
    else:                 # 135: diag up-right
        dx, dy = 1, -1

    H, W, C = arr.shape
    acc = np.zeros_like(arr, dtype=np.float32)
    cnt = 0

    # shift from -L//2 ... +L//2
    half = L // 2
    for t in range(-half, half + 1):
        sx = t * dx
        sy = t * dy

        # shift with padding (no wrap-around)
        shifted = np.zeros_like(arr, dtype=np.float32)

        x0_src = max(0, -sx)
        x1_src = min(W, W - sx)
        y0_src = max(0, -sy)
        y1_src = min(H, H - sy)

        x0_dst = max(0, sx)
        x1_dst = min(W, W + sx)
        y0_dst = max(0, sy)
        y1_dst = min(H, H + sy)

        if (x1_src > x0_src) and (y1_src > y0_src):
            shifted[y0_dst:y1_dst, x0_dst:x1_dst, :] = arr[y0_src:y1_src, x0_src:x1_src, :]
            acc += shifted
            cnt += 1

    if cnt > 0:
        out = acc / float(cnt)
    else:
        out = arr

    out = np.clip(out, 0.0, 1.0)
    return Image.fromarray((out * 255.0).astype(np.uint8))


def apply_noise(img: Image.Image, rng: random.Random) -> Image.Image:
    arr = np.array(img).astype(np.float32) / 255.0
    sigma = rng.uniform(0.005, 0.05)
    noise = rng.normalvariate(0, sigma)
    # gaussian noise per-pixel
    n = np.random.RandomState(rng.randrange(10**9)).normal(0, sigma, arr.shape).astype(np.float32)
    arr = np.clip(arr + n, 0.0, 1.0)
    return Image.fromarray((arr * 255.0).astype(np.uint8))


def apply_jpeg(img: Image.Image, rng: random.Random) -> Image.Image:
    import io
    q = int(rng.uniform(10, 60))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=q)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def apply_haze(img: Image.Image, rng: random.Random) -> Image.Image:
    # simple atmospheric veiling approximation: x' = x*(1-t) + A*t
    arr = np.array(img).astype(np.float32) / 255.0
    t = rng.uniform(0.05, 0.35)
    A = rng.uniform(0.7, 1.0)
    arr = arr * (1.0 - t) + A * t
    arr = np.clip(arr, 0.0, 1.0)
    return Image.fromarray((arr * 255.0).astype(np.uint8))


def apply_lowlight(img: Image.Image, rng: random.Random) -> Image.Image:
    factor = rng.uniform(0.3, 0.8)
    return ImageEnhance.Brightness(img).enhance(factor)


def apply_color_jitter(img: Image.Image, rng: random.Random) -> Image.Image:
    # mild color/contrast change
    c = rng.uniform(0.7, 1.3)
    s = rng.uniform(0.7, 1.3)
    img = ImageEnhance.Contrast(img).enhance(c)
    img = ImageEnhance.Color(img).enhance(s)
    return img


OPS = {
    "gaussian_blur": apply_gaussian_blur,
    "motion_blur": apply_motion_blur,
    "noise": apply_noise,
    "jpeg": apply_jpeg,
    "haze": apply_haze,
    "lowlight": apply_lowlight,
    "color_jitter": apply_color_jitter,
}


class Mixed7SynthDataset(Dataset):
    """
    Clean GT -> apply 1..7 random distortions on-the-fly
    Returns:
      inp_tensor, gt_tensor
    """

    def __init__(
        self,
        clean_root: str,
        mode: str = "train",
        transform=None,
        seed: int = 1234,
        min_ops: int = 1,
        max_ops: int = 7,
    ):
        assert mode in ["train", "test"]
        self.clean_root = clean_root
        self.mode = mode
        self.seed = int(seed)
        self.min_ops = int(min_ops)
        self.max_ops = int(max_ops)

        self.files = list_images(clean_root)

        self.transform = transform if transform is not None else T.Compose([
            T.Resize((256, 256)),
            T.ToTensor(),
        ])

        print(f"[Mixed7Synth] mode={mode} | clean={len(self.files)} | root={clean_root}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx: int):
        path = self.files[idx]
        gt_pil = Image.open(path).convert("RGB")

        # deterministic for test, stochastic for train (but seed-controlled)
        if self.mode == "test":
            rng = random.Random(self.seed + idx)
        else:
            rng = random.Random(self.seed * 1000003 + idx)

        # choose how many ops
        n_ops = rng.randint(self.min_ops, self.max_ops)

        # choose ops without replacement
        op_names = list(OPS.keys())
        rng.shuffle(op_names)
        chosen = op_names[:n_ops]

        inp_pil = gt_pil
        for op in chosen:
            inp_pil = OPS[op](inp_pil, rng)

        inp = self.transform(inp_pil)
        gt = self.transform(gt_pil)
        return inp, gt
