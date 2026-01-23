import os, sys, torch, torch.nn.functional as F, numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
from skimage.filters import sobel_h, sobel_v

# ----------------------------
# 경로 설정
# ----------------------------
ckpt_path = r"E:\restormer+volterra\checkpoints\restormer_volterra_rain100h\epoch_100.pth"
rain_path = r"E:\restormer+volterra\data\rain100L\train\rain\norain-7.png"
gt_path = r"E:\restormer+volterra\data\rain100L\train\norain\norain-7.png"
save_dir = r"E:\restormer+volterra\Restormer + Volterra\visualization"
os.makedirs(save_dir, exist_ok=True)

# ----------------------------
# 상위 폴더 등록
# ----------------------------
PROJECT_ROOT = r"E:\restormer+volterra\Restormer + Volterra"
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# ----------------------------
# 모델 로드
# ----------------------------
from models.restormer_volterra import RestormerVolterra
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = RestormerVolterra()
checkpoint = torch.load(ckpt_path, map_location=device)
model.load_state_dict(checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint)
model.to(device).eval()

# ----------------------------
# 이미지 로드
# ----------------------------
to_tensor = transforms.ToTensor()
inp = to_tensor(Image.open(rain_path).convert('RGB')).unsqueeze(0).to(device)
gt = to_tensor(Image.open(gt_path).convert('RGB')).unsqueeze(0).to(device)

# ----------------------------
# 모델 추론
# ----------------------------
with torch.no_grad():
    with torch.amp.autocast('cuda'):
        restored = model(inp)

restored = restored.float().clamp(0, 1).cpu()[0]   # ✅ float16 → float32 강제 변환
inp_cpu = inp.cpu()[0].float()
gt_cpu = gt.cpu()[0].float()

# ----------------------------
# Rain Layer 계산
# ----------------------------
rain_layer = torch.abs(inp_cpu - restored)

# ----------------------------
# Section Line (중앙 수평선)
# ----------------------------
y = restored.shape[1] // 2
section_inp = inp_cpu[:, y, :].mean(0).numpy()
section_out = restored[:, y, :].mean(0).numpy()
section_gt = gt_cpu[:, y, :].mean(0).numpy()

# ----------------------------
# Gradient Distribution
# ----------------------------
def gradient_histogram(img_tensor):
    img_gray = img_tensor.mean(0).numpy().astype(np.float32)
    grad_h = sobel_h(img_gray)
    grad_v = sobel_v(img_gray)
    return grad_h.flatten(), grad_v.flatten()

grad_h, grad_v = gradient_histogram(restored)

# ----------------------------
# Figure 생성
# ----------------------------
fig, axes = plt.subplots(4, 3, figsize=(15, 10))

titles = ['Input (Rainy)', 'ReVolT (ours)', 'Ground Truth']
imgs = [inp_cpu, restored, gt_cpu]

# 1행: Scene
for i in range(3):
    img_np = imgs[i].permute(1, 2, 0).numpy().astype(np.float32)  # ✅ dtype 변환
    axes[0, i].imshow(img_np)
    axes[0, i].set_title(titles[i], fontsize=11)
    axes[0, i].axis('off')

# 2행: Rain Layer
rain_np = rain_layer.permute(1, 2, 0).numpy().astype(np.float32)
axes[1, 0].imshow(np.zeros_like(rain_np))
axes[1, 1].imshow(rain_np)
axes[1, 2].imshow(np.zeros_like(rain_np))
axes[1, 1].set_title("Extracted Rain Layer")
for i in range(3): axes[1, i].axis('off')

# 3행: Section Line
axes[2, 0].plot(section_inp, color='gray', label='Rainy')
axes[2, 1].plot(section_out, color='red', label='ReVolT')
axes[2, 2].plot(section_gt, color='green', label='GT')
for i in range(3):
    axes[2, i].legend()
    axes[2, i].set_ylabel('Intensity')
    axes[2, i].set_xlabel('Width')

# 4행: Gradient Distribution
axes[3, 1].hist(grad_h, bins=100, density=True, color='r', alpha=0.5, label='Horizontal')
axes[3, 1].hist(grad_v, bins=100, density=True, color='b', alpha=0.5, label='Vertical')
axes[3, 1].set_xlim(-0.1, 0.1)
axes[3, 1].set_title('Gradient Distribution')
axes[3, 1].legend()
for i in [0, 2]:
    axes[3, i].axis('off')

plt.tight_layout()
save_path = os.path.join(save_dir, "derain_analysis_revolt.png")
plt.savefig(save_path, dpi=300)
plt.show()

print(f"✅ Figure saved to: {save_path}")
