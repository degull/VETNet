# E:\restormer+volterra\re_dataset\csd_dataset.py
import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import natsort


class CSDDataset(Dataset):
    """
    CSD (Desnowing) dataset loader.

    Folder structure (as you used before):
      E:/restormer+volterra/data/CSD/
        Train/
          Snow/   (*.png, *.jpg, *.tif, ...)
          Gt/     (*.png, *.jpg, *.tif, ...)
          Mask/   (optional, ignored)
        Test/
          Snow/
          Gt/
          Mask/   (optional, ignored)

    Returns:
      snow_img (Tensor), gt_img (Tensor)
    """

    IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

    def __init__(self, root_dir, split="train", transform=None, strict=True):
        """
        Args:
            root_dir (str): e.g., E:/restormer+volterra/data/CSD
            split (str): 'train' or 'test'
            transform: torchvision transform (applies to both Snow and Gt)
            strict (bool): if True, drop samples where GT pair is missing
        """
        assert split in ["train", "test"], "split must be 'train' or 'test'"

        self.root_dir = root_dir
        self.split = split

        # CSD uses Train/Test (capitalized)
        split_dir = "Train" if split == "train" else "Test"

        self.snow_dir = os.path.join(root_dir, split_dir, "Snow")
        self.gt_dir = os.path.join(root_dir, split_dir, "Gt")

        if not os.path.isdir(self.snow_dir):
            raise FileNotFoundError(f"snow_dir not found: {self.snow_dir}")
        if not os.path.isdir(self.gt_dir):
            raise FileNotFoundError(f"gt_dir not found: {self.gt_dir}")

        self.transform = transform if transform else transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])

        snow_files = [f for f in os.listdir(self.snow_dir) if f.lower().endswith(self.IMG_EXTS)]
        gt_files = [f for f in os.listdir(self.gt_dir) if f.lower().endswith(self.IMG_EXTS)]

        snow_files = natsort.natsorted(snow_files)
        gt_files = natsort.natsorted(gt_files)

        # Build GT index by stem
        gt_index = {}
        for f in gt_files:
            stem = os.path.splitext(f)[0]
            gt_index[stem] = os.path.join(self.gt_dir, f)

        pairs = []
        missing = 0

        for sf in snow_files:
            snow_path = os.path.join(self.snow_dir, sf)
            snow_stem = os.path.splitext(sf)[0]

            # Default pairing: same stem
            gt_path = gt_index.get(snow_stem, None)

            # Fallback: if any weird suffix exists, try base before first '_'
            if gt_path is None:
                base = snow_stem.split("_")[0]
                gt_path = gt_index.get(base, None)

            if gt_path is None:
                missing += 1
                if not strict:
                    pairs.append((snow_path, None))
                continue

            pairs.append((snow_path, gt_path))

        if len(pairs) == 0:
            raise RuntimeError(
                f"No valid (Snow, Gt) pairs found.\n"
                f"snow_dir={self.snow_dir}\n"
                f"gt_dir={self.gt_dir}\n"
                f"Missing GT for {missing} snow images."
            )

        self.pairs = pairs
        self.missing = missing

        print(f"[CSD] split={split} | pairs={len(self.pairs)} | missing_gt={self.missing}")
        print(f"[CSD] snow_dir: {self.snow_dir}")
        print(f"[CSD] gt_dir  : {self.gt_dir}")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        snow_path, gt_path = self.pairs[idx]
        if gt_path is None:
            raise FileNotFoundError(f"GT not found for snow: {snow_path}")

        snow = Image.open(snow_path).convert("RGB")
        gt = Image.open(gt_path).convert("RGB")

        snow = self.transform(snow)
        gt = self.transform(gt)

        # 🔍 디버깅용 (필요하면 주석 해제)
        # print(f"[{idx}] snow: {snow_path}")
        # print(f"[{idx}] gt  : {gt_path}")

        return snow, gt
