# E:\restormer+volterra\re_dataset\gopro_dataset.py
import os
import csv
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T


class GoProDataset(Dataset):
    """
    GoPro CSV loader

    CSV format:
      dist_img,ref_img
      E:/.../blur/xxx.png, E:/.../sharp/xxx.png
    or
      C:/.../blur/xxx.png, C:/.../sharp/xxx.png
    or relative paths (then resolved relative to csv directory)
    """

    def __init__(self, csv_file, transform=None, strict=True, skip_header=True):
        self.csv_file = csv_file
        self.csv_dir = os.path.dirname(os.path.abspath(csv_file))
        self.strict = strict

        self.transform = transform or T.Compose([
            T.Resize((256, 256)),
            T.ToTensor()
        ])

        pairs = []
        with open(csv_file, "r", newline="") as f:
            reader = csv.reader(f)

            for i, row in enumerate(reader):
                # 빈 줄 / 이상한 줄 방어
                if not row or len(row) < 2:
                    continue

                a, b = row[0].strip(), row[1].strip()

                # 헤더 자동 스킵
                if skip_header and i == 0:
                    # dist_img,ref_img 또는 blur,sharp 같은 헤더면 스킵
                    low0, low1 = a.lower(), b.lower()
                    if ("dist" in low0 or "blur" in low0) and ("ref" in low1 or "sharp" in low1 or "gt" in low1):
                        continue

                blur_path = self._resolve_path(a)
                sharp_path = self._resolve_path(b)

                if os.path.exists(blur_path) and os.path.exists(sharp_path):
                    pairs.append((blur_path, sharp_path))
                else:
                    if strict:
                        raise FileNotFoundError(
                            f"[GoProDataset] File not found at row {i}:\n"
                            f"  blur : {blur_path}\n"
                            f"  sharp: {sharp_path}\n"
                            f"  (raw) : {a} , {b}\n"
                            f"  csv   : {csv_file}"
                        )
                    # strict=False면 그냥 스킵
                    continue

        if len(pairs) == 0:
            raise RuntimeError(
                f"[GoProDataset] No valid pairs found.\n"
                f"csv_file={csv_file}\n"
                f"Try strict=False to skip missing files."
            )

        self.pairs = pairs
        print(f"[GoPro] pairs={len(self.pairs)} | strict={self.strict} | csv={self.csv_file}")

    def _resolve_path(self, p: str) -> str:
        """
        Resolve path robustly for Windows.
        - normalize slashes
        - if relative, resolve relative to csv directory
        - return absolute normalized path
        """
        p = p.strip().strip('"').strip("'")
        p = p.replace("\\", "/")  # unify
        # 절대경로면 그대로
        if os.path.isabs(p):
            return os.path.normpath(p)
        # 상대경로면 csv 위치 기준
        return os.path.normpath(os.path.join(self.csv_dir, p))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        blur_path, sharp_path = self.pairs[idx]
        blur = Image.open(blur_path).convert("RGB")
        sharp = Image.open(sharp_path).convert("RGB")
        return self.transform(blur), self.transform(sharp)
