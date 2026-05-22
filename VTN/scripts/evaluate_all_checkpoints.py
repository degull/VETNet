# evaluate all checkpoints
""" import os
import torch
import torchvision.transforms as transforms
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, ConcatDataset
from torch.cuda.amp import autocast
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim

from PIL import Image
from restormer_volterra import RestormerVolterra
from re_dataset.rain100l_dataset import Rain100LDataset
from re_dataset.hide_dataset import HIDEDataset

# ✅ 설정
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CHECKPOINT_DIR = r"E:\restormer+volterra\checkpoints\restormer_volterra_train_4sets"
START_EPOCH = 1
END_EPOCH = 100

# ✅ 테스트셋 경로
RAIN100L_DIR = 'E:/restormer+volterra/data/rain100L/test'
HIDE_DIR = 'E:/restormer+volterra/data/HIDE'

# ✅ 전처리
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# ✅ 테스트셋 로딩
rain100l_dataset = Rain100LDataset(root_dir=RAIN100L_DIR, transform=transform)
hide_dataset = HIDEDataset(root_dir=HIDE_DIR, transform=transform)
test_dataset = ConcatDataset([rain100l_dataset, hide_dataset])
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# ✅ 결과 저장용 리스트
results = []

# ✅ 반복 평가
for epoch in range(START_EPOCH, END_EPOCH + 1):
    ckpt_path = os.path.join(CHECKPOINT_DIR, f"epoch_{epoch}.pth")
    if not os.path.exists(ckpt_path):
        print(f"❌ Checkpoint not found: {ckpt_path}")
        continue

    # 모델 로딩
    model = RestormerVolterra().to(DEVICE)
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    model.eval()

    total_psnr = 0.0
    total_ssim = 0.0
    num_images = len(test_loader)

    with torch.no_grad():
        for distorted, reference in tqdm(test_loader, desc=f"Epoch {epoch}"):
            distorted = distorted.to(DEVICE)
            reference = reference.to(DEVICE)

            with autocast():
                output = model(distorted)

            output_np = output.squeeze(0).cpu().numpy().transpose(1, 2, 0)
            reference_np = reference.squeeze(0).cpu().numpy().transpose(1, 2, 0)

            output_np = np.clip(output_np, 0, 1)
            reference_np = np.clip(reference_np, 0, 1)

            psnr = compute_psnr(reference_np, output_np, data_range=1.0)
            ssim = compute_ssim(reference_np, output_np, data_range=1.0, channel_axis=2)

            total_psnr += psnr
            total_ssim += ssim

    avg_psnr = total_psnr / num_images
    avg_ssim = total_ssim / num_images
    results.append((epoch, avg_psnr, avg_ssim))
    print(f"✅ Epoch {epoch:3d} | PSNR: {avg_psnr:.2f} dB | SSIM: {avg_ssim:.4f}")

# ✅ 최고 성능 에폭 출력
best = max(results, key=lambda x: (x[1], x[2]))
print(f"\n🏆 Best Epoch: {best[0]} | PSNR: {best[1]:.2f} | SSIM: {best[2]:.4f}")
 """

# evaluate single checkpoint
import os
import torch
import torchvision.transforms as transforms
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, ConcatDataset
from torch.cuda.amp import autocast
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim
from PIL import Image

from models.restormer_volterra import RestormerVolterra
from re_dataset.rain100l_dataset import Rain100LDataset
from re_dataset.hide_dataset import HIDEDataset

# ✅ 설정
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CKPT_PATH = r"E:\restormer+volterra\checkpoints\restormer_volterra_train_4sets\epoch_97.pth"

# ✅ 테스트셋 경로
RAIN100L_DIR = 'E:/restormer+volterra/data/rain100L/test'
HIDE_DIR = 'E:/restormer+volterra/data/HIDE'

# ✅ 전처리
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# ✅ 테스트셋 로딩
rain100l_dataset = Rain100LDataset(root_dir=RAIN100L_DIR, transform=transform)
hide_dataset = HIDEDataset(root_dir=HIDE_DIR, transform=transform)
test_dataset = ConcatDataset([rain100l_dataset, hide_dataset])
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# ✅ 모델 로딩
assert os.path.exists(CKPT_PATH), f"❌ Checkpoint not found: {CKPT_PATH}"
print(f"🔁 Loading model from {CKPT_PATH}")
model = RestormerVolterra().to(DEVICE)
model.load_state_dict(torch.load(CKPT_PATH, map_location=DEVICE))
model.eval()

# ✅ 평가
total_psnr = 0.0
total_ssim = 0.0
num_images = len(test_loader)

with torch.no_grad():
    for distorted, reference in tqdm(test_loader, desc=f"Evaluating epoch_97.pth"):
        distorted = distorted.to(DEVICE)
        reference = reference.to(DEVICE)

        with autocast():
            output = model(distorted)

        output_np = output.squeeze(0).cpu().numpy().transpose(1, 2, 0)
        reference_np = reference.squeeze(0).cpu().numpy().transpose(1, 2, 0)

        output_np = np.clip(output_np, 0, 1)
        reference_np = np.clip(reference_np, 0, 1)

        psnr = compute_psnr(reference_np, output_np, data_range=1.0)
        ssim = compute_ssim(reference_np, output_np, data_range=1.0, channel_axis=2)

        total_psnr += psnr
        total_ssim += ssim

# ✅ 결과 출력
avg_psnr = total_psnr / num_images
avg_ssim = total_ssim / num_images
print(f"\n✅ [Evaluation Result for epoch_97.pth]")
print(f"📈 Average PSNR: {avg_psnr:.2f} dB")
print(f"📊 Average SSIM: {avg_ssim:.4f}")


# 🏆 Best Epoch: 100 | PSNR: 28.75 | SSIM: 0.8696