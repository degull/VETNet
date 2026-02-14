# E:\restormer+volterra\re_dataset\kadis700k_dataset.py
import os
import csv
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class KADIS700KCSVDataset(Dataset):
    """
    KADIS-700K (Mixed distortions) paired dataset loader using CSV.

    CSV format (header optional):
      dist_img,ref_img
      /abs/or/rel/path/to/dist.png,/abs/or/rel/path/to/ref.png
      ...

    - If CSV paths are absolute: uses as-is
    - If CSV paths are relative: it will be joined with root_dir
    - Optional remap: if CSV was created on another machine, you can remap prefix
        e.g., old_prefix="C:/Users/.../KADIS", new_prefix="E:/restormer+volterra/data/KADIS-700K"
    """

    def __init__(
        self,
        root_dir: str,
        csv_path: str,
        transform=None,
        strict: bool = True,
        old_prefix: str = "",
        new_prefix: str = "",
    ):
        self.root_dir = root_dir
        self.csv_path = csv_path
        self.strict = strict

        self.transform = transform if transform else transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])

        self.pairs = []
        missing = 0

        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV not found: {csv_path}")

        with open(csv_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            rows = list(reader)

        # header 처리(있으면 skip)
        if len(rows) > 0 and len(rows[0]) >= 2:
            if rows[0][0].lower() in ["dist_img", "distorted", "input"] and rows[0][1].lower() in ["ref_img", "gt", "target"]:
                rows = rows[1:]

        for r in rows:
            if len(r) < 2:
                continue
            dist_path, ref_path = r[0].strip(), r[1].strip()

            # prefix remap (optional)
            if old_prefix and new_prefix:
                dist_path = dist_path.replace(old_prefix, new_prefix)
                ref_path = ref_path.replace(old_prefix, new_prefix)

            # 상대경로면 root_dir 기준으로 붙임
            if not os.path.isabs(dist_path):
                dist_path = os.path.join(root_dir, dist_path)
            if not os.path.isabs(ref_path):
                ref_path = os.path.join(root_dir, ref_path)

            dist_path = os.path.normpath(dist_path)
            ref_path = os.path.normpath(ref_path)

            if strict:
                if (not os.path.exists(dist_path)) or (not os.path.exists(ref_path)):
                    missing += 1
                    continue

            self.pairs.append((dist_path, ref_path))

        if len(self.pairs) == 0:
            raise RuntimeError(
                f"No valid pairs found.\n"
                f"root_dir={root_dir}\n"
                f"csv={csv_path}\n"
                f"Check CSV paths (absolute/relative) and remap prefixes."
            )

        print(f"[KADIS-700K] pairs={len(self.pairs)} | missing={missing} | strict={strict}")
        print(f"[KADIS-700K] csv: {csv_path}")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        dist_path, ref_path = self.pairs[idx]
        dist = Image.open(dist_path).convert("RGB")
        ref = Image.open(ref_path).convert("RGB")
        return self.transform(dist), self.transform(ref)
