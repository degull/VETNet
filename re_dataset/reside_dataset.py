# E:\restormer+volterra\re_dataset\reside_dataset.py
import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import natsort


class RESIDEDataset(Dataset):
    """
    RESIDE-6K (haze removal) dataset loader.

    Folder structure:
      E:/restormer+volterra/data/RESIDE-6K/
        train/
          hazy/   (*.jpg, *.png, ...)
          GT/     (*.jpg, *.png, ...)
        test/
          hazy/
          GT/

    Notes:
    - Many RESIDE/ITS hazy filenames include extra suffixes (e.g., _0.8_0.2).
      We map hazy -> GT using "base name before first '_'".
      Example:
        hazy:  0001_0.8_0.2.jpg
        GT:    0001.jpg  (or 0001.png)
    - If your dataset uses 1-to-1 identical filenames (no suffix), it will still work.
    """

    IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

    def __init__(self, root_dir, split="train", transform=None, strict=True):
        """
        Args:
            root_dir (str): e.g., E:/restormer+volterra/data/RESIDE-6K
            split (str): 'train' or 'test'
            transform: torchvision transform (applies to both hazy and GT)
            strict (bool): if True, drop hazy images whose GT cannot be found
        """
        assert split in ["train", "test"], "split must be 'train' or 'test'"

        self.root_dir = root_dir
        self.split = split
        self.hazy_dir = os.path.join(root_dir, split, "hazy")
        self.gt_dir = os.path.join(root_dir, split, "GT")

        if not os.path.isdir(self.hazy_dir):
            raise FileNotFoundError(f"hazy_dir not found: {self.hazy_dir}")
        if not os.path.isdir(self.gt_dir):
            raise FileNotFoundError(f"gt_dir not found: {self.gt_dir}")

        self.transform = transform if transform else transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])

        # Build GT index: key = stem (filename without ext), value = full path
        gt_files = [f for f in os.listdir(self.gt_dir) if f.lower().endswith(self.IMG_EXTS)]
        gt_files = natsort.natsorted(gt_files)

        self.gt_index = {}
        for f in gt_files:
            stem = os.path.splitext(f)[0]
            self.gt_index[stem] = os.path.join(self.gt_dir, f)

        # Build pairs from hazy
        hazy_files = [f for f in os.listdir(self.hazy_dir) if f.lower().endswith(self.IMG_EXTS)]
        hazy_files = natsort.natsorted(hazy_files)

        pairs = []
        missing = 0

        for hf in hazy_files:
            hazy_path = os.path.join(self.hazy_dir, hf)
            hazy_stem = os.path.splitext(hf)[0]

            # Common RESIDE mapping: take base before first '_' as GT stem
            # e.g., 0001_0.8_0.2 -> 0001
            gt_stem = hazy_stem.split("_")[0]
            gt_path = self.gt_index.get(gt_stem, None)

            # Fallback: sometimes hazy has same stem as GT
            if gt_path is None and hazy_stem in self.gt_index:
                gt_path = self.gt_index[hazy_stem]

            if gt_path is None:
                missing += 1
                if not strict:
                    # keep but will error on __getitem__ (not recommended)
                    pairs.append((hazy_path, None))
                continue
            pairs.append((hazy_path, gt_path))

        if len(pairs) == 0:
            raise RuntimeError(
                f"No valid (hazy, GT) pairs found.\n"
                f"hazy_dir={self.hazy_dir}\n"
                f"gt_dir={self.gt_dir}\n"
                f"Missing GT for {missing} hazy images."
            )

        self.pairs = pairs
        self.missing = missing

        print(f"[RESIDE] split={split} | pairs={len(self.pairs)} | missing_gt={self.missing}")
        print(f"[RESIDE] hazy_dir: {self.hazy_dir}")
        print(f"[RESIDE] gt_dir  : {self.gt_dir}")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        hazy_path, gt_path = self.pairs[idx]
        if gt_path is None:
            raise FileNotFoundError(f"GT not found for hazy: {hazy_path}")

        hazy = Image.open(hazy_path).convert("RGB")
        gt = Image.open(gt_path).convert("RGB")

        hazy = self.transform(hazy)
        gt = self.transform(gt)

        # 🔍 디버깅용 (필요하면 주석 해제)
        # print(f"[{idx}] hazy: {hazy_path}")
        # print(f"[{idx}] gt  : {gt_path}")

        return hazy, gt
