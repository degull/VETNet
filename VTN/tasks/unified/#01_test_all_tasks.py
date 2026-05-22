""" import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import pandas as pd
import numpy as np
from glob import glob
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim

from models.restormer_volterra import RestormerVolterra


class PairedFolderDataset(Dataset):
    def __init__(self, input_dir, target_dir, transform):
        self.input_paths = sorted(glob(os.path.join(input_dir, "**", "*.*"), recursive=True))
        self.input_paths = [f for f in self.input_paths if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"))]

        # GT를 확장자 제거 후 이름만 매칭되도록 수정
        gt_files = {os.path.splitext(f)[0]: os.path.join(target_dir, f)
                    for f in os.listdir(target_dir)
                    if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"))}

        self.target_paths = []
        for inp_path in self.input_paths:
            base = os.path.splitext(os.path.basename(inp_path))[0]
            if base in gt_files:
                self.target_paths.append(gt_files[base])
            else:
                raise FileNotFoundError(f"GT 파일이 없음 (입력: {inp_path}, base: {base})")

        self.transform = transform

        assert len(self.input_paths) == len(self.target_paths), \
            f"Mismatched input and GT lengths: {len(self.input_paths)} vs {len(self.target_paths)}"

    def __len__(self):
        return len(self.input_paths)

    def __getitem__(self, idx):
        inp = Image.open(self.input_paths[idx]).convert("RGB")
        tgt = Image.open(self.target_paths[idx]).convert("RGB")
        return self.transform(inp), self.transform(tgt)


class PairedCSVDataset(Dataset):
    def __init__(self, csv_path, transform):
        df = pd.read_csv(csv_path)

        # ✅ 열 이름이 'dist_img', 'ref_img'인 경우 처리
        if 'dist_img' in df.columns and 'ref_img' in df.columns:
            self.paths = df[['dist_img', 'ref_img']].values.tolist()
        else:
            self.paths = df.values.tolist()

        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        inp_path, tgt_path = self.paths[idx]
        inp = Image.open(inp_path).convert("RGB")
        tgt = Image.open(tgt_path).convert("RGB")
        return self.transform(inp), self.transform(tgt)



def evaluate(model, dataloader, name):
    model.eval()
    psnr_total = 0
    ssim_total = 0
    count = 0

    with torch.no_grad():
        for inputs, targets in tqdm(dataloader, desc=f"[{name}]"):
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs).clamp(0, 1)

            outputs = outputs.cpu().numpy()
            targets = targets.cpu().numpy()
            for out, gt in zip(outputs, targets):
                out = np.transpose(out, (1, 2, 0))
                gt = np.transpose(gt, (1, 2, 0))
                psnr_total += compute_psnr(gt, out, data_range=1.0)
                ssim_total += compute_ssim(gt, out, data_range=1.0, channel_axis=-1)
                count += 1

    print(f"📌 {name:12s} → PSNR: {psnr_total/count:.2f} dB, SSIM: {ssim_total/count:.4f}")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RestormerVolterra().to(device)

    checkpoint = r"E:\restormer+volterra\checkpoints\#01_all_tasks_balanced_160\epoch_100_ssim0.9177_psnr32.58.pth"
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    print(f"\n✅ Loaded checkpoint from {checkpoint}\n")

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    test_sets = {
        "Rain100H": PairedFolderDataset(
            r"E:/restormer+volterra/data/rain100H/test/rain",
            r"E:/restormer+volterra/data/rain100H/test/norain", transform),
        "Rain100L": PairedFolderDataset(
            r"E:/restormer+volterra/data/rain100L/test/rain",
            r"E:/restormer+volterra/data/rain100L/test/norain", transform),
        "HIDE": PairedFolderDataset(
            r"E:/restormer+volterra/data/HIDE/test",
            r"E:/restormer+volterra/data/HIDE/GT", transform),
        "SIDD": PairedCSVDataset(
            r"E:/restormer+volterra/data/SIDD/sidd_test_pairs.csv", transform),
        "CSD": PairedFolderDataset(
            r"E:/restormer+volterra/data/CSD/Test/Snow",
            r"E:/restormer+volterra/data/CSD/Test/Gt", transform),
    }

    for name, dataset in test_sets.items():
        loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)
        evaluate(model, loader, name)
 """

# 다른 데이터셋 테스트
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import pandas as pd
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim

from models.restormer_volterra import RestormerVolterra

class PairedListDataset(Dataset):
    """
    TXT 목록에서 (input, target) 경로 쌍을 읽는다.
    - 한 줄 예시:
        E:/.../blur/level_1/I01_01_01.png  E:/.../gt/I01.png
        E:/.../impulse/level_2/I02_13_02.png,E:/.../gt/I02.png
    - 공백 또는 콤마로 구분 가능
    - 주석(#) 라인은 무시
    """
    def __init__(self, list_path, transform):
        self.transform = transform
        base_dir = os.path.dirname(list_path)
        pairs = []

        with open(list_path, "r", encoding="utf-8") as f:
            lines = [ln.strip() for ln in f if ln.strip() and not ln.strip().startswith("#")]

        for ln in lines:
            parts = [p.strip().strip('"').strip("'") for p in (ln.split(",") if "," in ln else ln.split())]
            if len(parts) < 2:
                continue
            a, b = parts[0], parts[1]
            if not os.path.isabs(a):
                a = os.path.normpath(os.path.join(base_dir, a))
            if not os.path.isabs(b):
                b = os.path.normpath(os.path.join(base_dir, b))
            if os.path.exists(a) and os.path.exists(b):
                pairs.append((a, b))

        if len(pairs) == 0:
            raise FileNotFoundError(f"No valid pairs in {list_path}")

        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        a, b = self.pairs[idx]
        inp = Image.open(a).convert("RGB")
        tgt = Image.open(b).convert("RGB")
        return self.transform(inp), self.transform(tgt)


# ------------------ SOTS Dataset ------------------
class SOTSDataset(torch.utils.data.Dataset):
    def __init__(self, hazy_dir, clear_dir, transform):
        self.hazy_paths = sorted([
            os.path.join(hazy_dir, f)
            for f in os.listdir(hazy_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ])
        self.clear_dir = clear_dir
        self.transform = transform

    def __len__(self):
        return len(self.hazy_paths)

    def __getitem__(self, idx):
        hazy_path = self.hazy_paths[idx]
        fname = os.path.basename(hazy_path)

        # hazy 파일명: 1400_1.png → GT 파일명: 1400.png
        base = fname.split("_")[0] + ".png"
        clear_path = os.path.join(self.clear_dir, base)

        if not os.path.exists(clear_path):
            raise FileNotFoundError(f"GT 파일이 없음 (입력: {hazy_path}, 예상 GT: {clear_path})")

        hazy = Image.open(hazy_path).convert("RGB")
        clear = Image.open(clear_path).convert("RGB")

        return self.transform(hazy), self.transform(clear)


# ------------------ CSV 기반 Dataset ------------------
class PairedCSVDataset(Dataset):
    def __init__(self, file_path, transform):
        # 확장자 확인
        ext = os.path.splitext(file_path)[1].lower()
        if ext in [".csv", ".txt"]:
            try:
                df = pd.read_csv(file_path)
                if df.shape[1] >= 2:   # 2개 이상의 열 (dist, ref)
                    if 'dist_img' in df.columns and 'ref_img' in df.columns:
                        self.paths = df[['dist_img', 'ref_img']].values.tolist()
                    else:
                        self.paths = df.iloc[:, :2].values.tolist()
                else:
                    raise ValueError(f"{file_path} has only one column (need dist & ref)")
            except Exception as e:
                raise ValueError(f"⚠️ 파일을 읽는 중 에러 발생: {file_path}\n{e}")
        else:
            raise ValueError(f"지원하지 않는 파일 형식: {file_path}")

        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        inp_path, tgt_path = self.paths[idx]
        inp = Image.open(inp_path).convert("RGB")
        tgt = Image.open(tgt_path).convert("RGB")
        return self.transform(inp), self.transform(tgt)



# ------------------ 평가 함수 ------------------
def evaluate(model, dataloader, name):
    model.eval()
    psnr_total = 0
    ssim_total = 0
    count = 0

    with torch.no_grad():
        for inputs, targets in tqdm(dataloader, desc=f"[{name}]"):
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs).clamp(0, 1)

            outputs = outputs.cpu().numpy()
            targets = targets.cpu().numpy()
            for out, gt in zip(outputs, targets):
                out = np.transpose(out, (1, 2, 0))
                gt = np.transpose(gt, (1, 2, 0))
                psnr_total += compute_psnr(gt, out, data_range=1.0)
                ssim_total += compute_ssim(gt, out, data_range=1.0, channel_axis=-1)
                count += 1

    print(f"📌 {name:12s} → PSNR: {psnr_total/count:.2f} dB, SSIM: {ssim_total/count:.4f}")


# ------------------ 실행 ------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RestormerVolterra().to(device)

    checkpoint = r"E:\restormer+volterra\checkpoints\#01_all_tasks_balanced_160\epoch_100_ssim0.9177_psnr32.58.pth"
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    print(f"\n✅ Loaded checkpoint from {checkpoint}\n")

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    # ------------------ 테스트셋 정의 ------------------
    test_sets = {
        "SOTS-indoor": SOTSDataset(
            r"E:\restormer+volterra\data\SOTS\indoor\hazy",
            r"E:\restormer+volterra\data\SOTS\indoor\clear",
            transform
        ),
        "SOTS-outdoor": SOTSDataset(
            r"E:\restormer+volterra\data\SOTS\outdoor\hazy",
            r"E:\restormer+volterra\data\SOTS\outdoor\clear",
            transform
        ),

        # ✅ 이제 txt 기반으로 직접 매칭
        "KADID-gaussian": PairedListDataset(
            r"E:\restormer+volterra\data\kadid_seperate\gaussian\pairs_gaussian.txt",
            transform
        ),
        "KADID-impulse": PairedListDataset(
            r"E:\restormer+volterra\data\kadid_seperate\impulse noise\pairs_impulse.txt",
            transform
        ),
        "KADID-white": PairedListDataset(
            r"E:\restormer+volterra\data\kadid_seperate\white noise\pairs_white.txt",
            transform
        )
    }



    # ------------------ 평가 실행 ------------------
    for name, dataset in test_sets.items():
        loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)
        evaluate(model, loader, name)
