import torch
from torchvision.io.image import decode_jpeg, encode_jpeg
from torchvision import transforms
import numpy as np
import random
import math
from torch.nn import functional as F
import io
from PIL import Image
import kornia

from .utils_distortions import fspecial, filter2D, curves, imscatter, mapmm


def gaussian_blur(x: torch.Tensor, blur_sigma: int = 0.1) -> torch.Tensor:
    fs = 2 * math.ceil(2 * blur_sigma) + 1
    h = fspecial('gaussian', (fs, fs), blur_sigma)
    h = torch.from_numpy(h).float().to(x.device)
    if len(x.shape) == 3:
        x = x.unsqueeze(0)
    return filter2D(x, h.unsqueeze(0)).squeeze(0)


def lens_blur(x: torch.Tensor, radius: int) -> torch.Tensor:
    h = fspecial('disk', radius)
    h = torch.from_numpy(h).float().to(x.device)
    if len(x.shape) == 3:
        x = x.unsqueeze(0)
    return filter2D(x, h.unsqueeze(0)).squeeze(0)


def motion_blur(x: torch.Tensor, radius: int, angle: bool = None) -> torch.Tensor:
    if angle is None:
        angle = random.randint(0, 180)
    h = fspecial('motion', radius, angle)
    h = torch.from_numpy(h.copy()).float().to(x.device)
    if len(x.shape) == 3:
        x = x.unsqueeze(0)
    return filter2D(x, h.unsqueeze(0)).squeeze(0)


def color_diffusion(x: torch.Tensor, amount: int) -> torch.Tensor:
    blur_sigma = 1.5 * amount + 2
    scaling = amount
    x = x[[2, 1, 0], ...]
    lab = kornia.color.rgb_to_lab(x)
    fs = 2 * math.ceil(2 * blur_sigma) + 1
    h = fspecial('gaussian', (fs, fs), blur_sigma)
    h = torch.from_numpy(h).float().to(x.device)
    if len(lab.shape) == 3:
        lab = lab.unsqueeze(0)
    diff_ab = filter2D(lab[:, 1:3, ...], h.unsqueeze(0))
    lab[:, 1:3, ...] = diff_ab * scaling
    y = torch.trunc(kornia.color.lab_to_rgb(lab) * 255.) / 255.
    return y[:, [2, 1, 0]].squeeze(0)


def color_shift(x: torch.Tensor, amount: int) -> torch.Tensor:
    def perc(x, perc):
        xs, _ = torch.sort(x.flatten())
        i = len(xs) * perc / 100.
        i = max(min(i, len(xs)), 1)
        v = xs[round(i - 1)]
        return v

    gray = kornia.color.rgb_to_grayscale(x)
    gradxy = kornia.filters.spatial_gradient(gray.unsqueeze(0), 'diff')
    e = torch.sum(gradxy ** 2, 2) ** 0.5  # [1, H, W]

    fs = 2 * math.ceil(2 * 4) + 1
    h = fspecial('gaussian', (fs, fs), 4)
    h = torch.from_numpy(h).float().to(x.device)

    e = filter2D(e, h.unsqueeze(0))
    e = e.squeeze()  # ✅ [H, W]

    mine = torch.min(e)
    maxe = torch.max(e)
    if mine < maxe:
        e = (e - mine) / (maxe - mine)

    percdev = [1, 1]
    valuehi = perc(e, 100 - percdev[1])
    valuelo = 1 - perc(1 - e, 100 - percdev[0])
    e = torch.max(torch.min(e, valuehi), valuelo)

    # 채널 선택
    channel = 1
    g = x[channel, :, :]
    a = np.random.random((1, 2))
    amount_shift = np.round(a / (np.sum(a ** 2) ** 0.5) * amount)[0].astype(int)

    # shift
    y = F.pad(g, (amount_shift[0], amount_shift[0]), mode='replicate')
    y = F.pad(y.transpose(1, 0), (amount_shift[1], amount_shift[1]), mode='replicate').transpose(1, 0)
    y = torch.roll(y, (amount_shift[0], amount_shift[1]), dims=(0, 1))

    if amount_shift[1] != 0:
        y = y[amount_shift[1]:-amount_shift[1], ...]
    if amount_shift[0] != 0:
        y = y[..., amount_shift[0]:-amount_shift[0]]

    # ✅ e를 [H, W] → 자동 broadcast
    yblend = y * e + x[channel, ...] * (1 - e)
    x[channel, ...] = yblend

    return x



def color_saturation1(x: torch.Tensor, factor: int) -> torch.Tensor:
    x = x[[2, 1, 0], ...]
    hsv = kornia.color.rgb_to_hsv(x)
    hsv[1, ...] *= factor
    return kornia.color.hsv_to_rgb(hsv)[[2, 1, 0], ...]


def color_saturation2(x: torch.Tensor, factor: int) -> torch.Tensor:
    x = x[[2, 1, 0], ...]
    lab = kornia.color.rgb_to_lab(x)
    lab[1:3, ...] = lab[1:3, ...] * factor
    y = torch.trunc(kornia.color.lab_to_rgb(lab) * 255) / 255.
    return y[[2, 1, 0], ...]


def jpeg2000(x: torch.Tensor, ratio: int) -> torch.Tensor:
    device = x.device  # 원래 tensor의 device 보관
    x_cpu = (x * 255.).byte().cpu().numpy()
    pil_img = Image.fromarray(x_cpu.transpose(1, 2, 0), 'RGB')

    with io.BytesIO() as output:
        pil_img.save(output, format='JPEG2000')
        compressed_data = output.getvalue()

    y = Image.open(io.BytesIO(compressed_data))
    return transforms.ToTensor()(y).to(device)



def jpeg(x: torch.Tensor, quality: int) -> torch.Tensor:
    x *= 255.
    y = encode_jpeg(x.byte().cpu(), quality=quality)
    return (decode_jpeg(y) / 255.).to(torch.float32).to(x.device)


def white_noise(x: torch.Tensor, var: float, clip=True, rounds=False) -> torch.Tensor:
    noise = torch.randn(*x.size(), dtype=x.dtype, device=x.device) * math.sqrt(var)
    y = x + noise
    if clip:
        y = torch.clip(y, 0, 1)
    return y


def white_noise_cc(x: torch.Tensor, var: float, clip=True, rounds=False) -> torch.Tensor:
    noise = torch.randn(*x.size(), dtype=x.dtype, device=x.device) * math.sqrt(var)
    ycbcr = kornia.color.rgb_to_ycbcr(x)
    y = kornia.color.ycbcr_to_rgb(ycbcr + noise)
    if clip:
        y = torch.clip(y, 0, 1)
    return y


def impulse_noise(x: torch.Tensor, d: float, s_vs_p: float = 0.5) -> torch.Tensor:
    x = x.clone()
    num_sp = int(d * x.numel())
    coords = np.random.randint(0, x.shape[1], (num_sp, 2))
    for i in range(num_sp):
        c = np.random.randint(0, 3)
        if i < num_sp * s_vs_p:
            x[c, coords[i, 0], coords[i, 1]] = 1
        else:
            x[c, coords[i, 0], coords[i, 1]] = 0
    return x


def multiplicative_noise(x: torch.Tensor, var: float) -> torch.Tensor:
    noise = torch.randn(*x.size(), dtype=x.dtype, device=x.device) * math.sqrt(var)
    return torch.clip(x + x * noise, 0, 1)


def brighten(x: torch.Tensor, amount: float) -> torch.Tensor:
    x = x[[2, 1, 0]]
    lab = kornia.color.rgb_to_lab(x)
    l = lab[0, ...] / 100.
    l_ = curves(l, 0.5 + amount / 2)
    lab[0, ...] = l_ * 100.
    j = torch.clamp(kornia.color.lab_to_rgb(lab), 0, 1)
    return j[[2, 1, 0]]


def darken(x: torch.Tensor, amount: float) -> torch.Tensor:
    x = x[[2, 1, 0]]
    lab = kornia.color.rgb_to_lab(x)
    l = lab[0, ...] / 100.
    l_ = curves(l, 0.5 - amount / 2)
    lab[0, ...] = l_ * 100.
    j = torch.clamp(kornia.color.lab_to_rgb(lab), 0, 1)
    return j[[2, 1, 0]]


def mean_shift(x: torch.Tensor, amount: float) -> torch.Tensor:
    return torch.clamp(x + amount, 0, 1)


def jitter(x: torch.Tensor, amount: float) -> torch.Tensor:
    return imscatter(x, amount, 5)


def non_eccentricity_patch(x: torch.Tensor, pnum: int) -> torch.Tensor:
    y = x.clone()
    patch_size = [16, 16]
    for _ in range(pnum):
        py = random.randint(0, x.shape[1] - patch_size[0])
        px = random.randint(0, x.shape[2] - patch_size[1])
        patch = y[:, py:py + patch_size[0], px:px + patch_size[1]]
        y[:, py:py + patch_size[0], px:px + patch_size[1]] = patch
    return y


def pixelate(x: torch.Tensor, strength: float) -> torch.Tensor:
    c, h, w = x.shape
    z = 0.95 - strength ** 0.6
    ylo = kornia.geometry.resize(x, (int(h * z), int(w * z)), interpolation="nearest")
    return kornia.geometry.resize(ylo, (h, w), interpolation="nearest")


def quantization(x: torch.Tensor, levels: int) -> torch.Tensor:
    bins = torch.linspace(0, 1, levels, device=x.device)
    xq = torch.bucketize(x, bins) / levels
    return xq


def color_block(x: torch.Tensor, pnum: int) -> torch.Tensor:
    y = x.clone()
    for _ in range(pnum):
        py = random.randint(0, x.shape[1] - 32)
        px = random.randint(0, x.shape[2] - 32)
        color = torch.rand(3, 1, 1, device=x.device)
        y[:, py:py + 32, px:px + 32] = color
    return y


def high_sharpen(x: torch.Tensor, amount: int, radius: int = 3) -> torch.Tensor:
    x = x[[2, 1, 0]]
    lab = kornia.color.rgb_to_lab(x)
    l = lab[0:1, ...].unsqueeze(0)
    filt_radius = math.ceil(radius * 2)
    fs = 2 * filt_radius + 1
    h = fspecial('gaussian', (fs, fs), filt_radius)
    h = torch.from_numpy(h).float().to(x.device)
    sharp_filter = torch.zeros((fs, fs), device=x.device)
    sharp_filter[filt_radius, filt_radius] = 1
    sharp_filter -= h
    sharp_filter *= amount
    sharp_filter[filt_radius, filt_radius] += 1
    l = filter2D(l, sharp_filter.unsqueeze(0))
    lab[0, ...] = l.squeeze(0)
    return kornia.color.lab_to_rgb(lab)[[2, 1, 0], ...]


def linear_contrast_change(x: torch.Tensor, amount: float) -> torch.Tensor:
    return curves(x, [0.25 - amount / 4, 0.75 + amount / 4])


def non_linear_contrast_change(x: torch.Tensor, output_offset_value: float) -> torch.Tensor:
    return torch.clamp(x * (1 + output_offset_value), 0, 1)



