# VTN-IR: Volterra-Enhanced Transformer Network for Unified Image Restoration

<p align="center">
  A unified Transformer-based framework for deraining, deblurring, dehazing, and desnowing.
</p>

<p align="center">
  비·흐림·안개·눈 제거를 하나의 구조로 처리하는 통합 Transformer 기반 이미지 복원 프레임워크입니다.
</p>

<p align="center">
  <a href="#overview">English</a> · <a href="#한국어-소개">한국어</a> ·
  <a href="#results">Results</a> · <a href="#repository-structure">Code</a>
</p>

---

## Overview


<img width="1460" height="820" alt="vtn-ir-architecture" src="https://github.com/user-attachments/assets/05ac5ac7-4f59-4872-8ff7-28b81a04bf12" />


VTN-IR is a unified image restoration network that augments a hierarchical Transformer encoder-decoder with a **truncated second-order Volterra operator**. The proposed Volterra-Enhanced Transformer (VET) block explicitly models multiplicative cross-feature interactions that are otherwise learned only implicitly through attention, nonlinear activations, or gating.

The same shared network handles multiple degradation types without task-specific branches or conditional modules:

- Rain removal
- Motion deblurring
- Image dehazing
- Snow removal
- Unified all-in-one restoration
- Composite degradation restoration

### Key contributions

- **Explicit second-order interaction modeling:** VTN-IR combines a first-order convolutional response with a low-rank quadratic response.
- **Volterra-Enhanced Transformer block:** The Volterra operator is inserted after both the convolution attention pathway (MDTA) and the gated convolution feed-forward pathway (GDFN).
- **Unified restoration:** A single shared architecture handles heterogeneous and coexisting degradations without degradation-specific branches.
- **Rank-controlled formulation:** Low-rank factorization avoids explicitly constructing a full quadratic kernel. The main model uses rank \(R=4\).

Method

For an input feature map (X \in \mathbb{R}^{C \times H \times W}), the truncated second-order Volterra operator is defined as

L * X+\Gamma_2 \sum_{r=1}^{R}\left(Q_r^{(1)} * X\right)\odot\left(Q_r^{(2)} * X\right),$$

where (*) is convolution, (\odot) is element-wise multiplication, (R) is the interaction rank, and (\Gamma_2) controls the contribution of the quadratic response.

Each VET block applies this operator to both major Transformer pathways:

$$Z_1 = X + \mathcal{V}!\left(\mathrm{MDTA}(\mathrm{LN}(X))\right),$$

$$Z_2 = Z_1 + \mathcal{V}!\left(\mathrm{GDFN}(\mathrm{LN}(Z_1))\right).$$

The complete model follows a hierarchical encoder-latent-decoder design with skip addition and a final refinement stage. The default implementation uses base width 48, block depths ([4,6,6,8]), attention heads ([1,2,4,8]), four refinement blocks, and Volterra rank 4.

## Results

Results are reported as **PSNR / SSIM**.

### Single-task restoration

| Method | Rain100H | Rain100L | GoPro | RESIDE-6K | CSD |
|---|---:|---:|---:|---:|---:|
| Restormer | 31.46 / 0.9042 | 36.74 / 0.9786 | 32.92 / 0.9613 | 30.87 / 0.9694 | 35.43 / 0.9703 |
| PromptIR | 31.85 / 0.9107 | 37.04 / 0.9791 | 33.10 / 0.9634 | 31.31 / 0.9735 | 35.80 / 0.9724 |
| DiffUIR | 32.10 / 0.9183 | 37.45 / 0.9812 | 33.35 / 0.9656 | 31.55 / 0.9752 | 36.05 / 0.9741 |
| DA-CLIP | 32.35 / 0.9236 | 37.80 / 0.9824 | 33.60 / 0.9673 | 31.70 / 0.9771 | 36.30 / 0.9757 |
| AdaIR | 32.70 / 0.9304 | 38.90 / 0.9853 | 33.85 / 0.9695 | 31.80 / 0.9812 | 36.55 / 0.9776 |
| MoCE-IR | 33.10 / 0.9418 | 39.25 / 0.9861 | 34.10 / 0.9714 | 32.00 / 0.9823 | 36.80 / 0.9792 |
| MambaIRv2 | 33.80 / 0.9567 | 39.70 / 0.9874 | 34.35 / 0.9732 | 32.15 / 0.9828 | 37.20 / 0.9815 |
| HINT | 33.40 / 0.9485 | **40.04 / 0.9866** | 34.20 / 0.9727 | 32.24 / 0.9817 | 37.00 / 0.9806 |
| **VTN-IR (Ours)** | **34.47 / 0.9767** | 39.20 / 0.9843 | **34.55 / 0.9746** | **32.40 / 0.9834** | **37.35 / 0.9827** |

### Unified all-in-one restoration

A single shared model is trained on deraining, deblurring, dehazing, and desnowing.

| Method | Rain100H | GoPro | RESIDE-6K | CSD | Average |
|---|---:|---:|---:|---:|---:|
| Restormer | 31.55 / 0.9086 | 28.44 / 0.8920 | 22.83 / 0.8680 | 28.79 / 0.9143 | 27.91 / 0.8957 |
| PromptIR | 31.63 / 0.9041 | 29.00 / 0.8955 | 22.35 / 0.8628 | 27.75 / 0.9061 | 27.68 / 0.8921 |
| DiffUIR | 25.39 / 0.7739 | 29.93 / 0.8835 | 22.57 / 0.9097 | 24.29 / 0.8980 | 25.55 / 0.8663 |
| DA-CLIP | 30.80 / 0.8920 | 29.20 / 0.9000 | 24.30 / 0.9000 | 28.20 / 0.9100 | 28.13 / 0.9005 |
| AdaIR | 26.17 / 0.8122 | **33.66 / 0.9582** | 24.70 / 0.9036 | 23.20 / 0.8313 | 26.93 / 0.8763 |
| MoCE-IR | 30.14 / 0.8742 | 28.95 / 0.8900 | 26.20 / 0.9160 | 26.68 / 0.8740 | 27.99 / 0.8886 |
| MambaIRv2 | 31.20 / 0.9050 | 29.40 / 0.9020 | 26.50 / 0.9220 | 28.50 / 0.9120 | 28.90 / 0.9103 |
| HINT | 29.63 / 0.8676 | 27.92 / 0.8815 | 25.01 / 0.9090 | 26.32 / 0.8759 | 27.22 / 0.8835 |
| **VTN-IR (Ours)** | **31.80 / 0.9120** | 29.20 / 0.9030 | **26.80 / 0.9250** | **29.00 / 0.9160** | **29.20 / 0.9140** |

### Composite degradation restoration

| Method | Rain + Haze | Rain + Blur | Haze + Snow | Average |
|---|---:|---:|---:|---:|
| Restormer | 16.81 / 0.7371 | 13.82 / 0.4390 | 25.34 / 0.8895 | 18.66 / 0.6885 |
| PromptIR | 16.84 / 0.7388 | 13.91 / 0.4341 | 26.05 / 0.8880 | 18.93 / 0.6870 |
| DiffUIR | 19.45 / 0.6973 | 15.55 / 0.4627 | 23.21 / 0.8497 | 19.40 / 0.6699 |
| AdaIR | 19.84 / 0.6834 | 14.85 / 0.4415 | 23.20 / 0.8240 | 19.30 / 0.6496 |
| MoCE-IR | 16.50 / 0.6957 | 13.28 / 0.4387 | 25.75 / 0.8623 | 18.51 / 0.6656 |
| HINT | 16.67 / 0.6917 | 13.28 / 0.4644 | 25.74 / 0.8631 | 18.56 / 0.6731 |
| **VTN-IR (Ours)** | **20.35 / 0.7520** | **15.90 / 0.4850** | **26.40 / 0.9020** | **20.88 / 0.7130** |

## Datasets

| Task | Dataset |
|---|---|
| Deraining | Rain100H, Rain100L |
| Motion deblurring | GoPro |
| Dehazing | RESIDE-6K |
| Desnowing | CSD |
| Composite restoration | Rain + Haze, Rain + Blur, Haze + Snow |

Dataset files are not included in this repository. Download each dataset from its official source and configure its path under `VTN/data/` or update `VTN/config.py`.

## Environment

The code is implemented in Python and PyTorch. The main dependencies used throughout the repository include:

```text
torch
torchvision
numpy
Pillow
scikit-image
pandas
scipy
tqdm
kornia
natsort
thop
```

Exact package versions are not currently pinned. A CUDA-enabled PyTorch environment is recommended for training and evaluation.

## Repository structure

```text
VTN_IR/
├── VTN/
│   ├── models/                 # VTN-IR and Volterra operator
│   ├── scripts/                # Training, evaluation, ablation, and benchmark scripts
│   ├── tasks/                  # Task-specific scripts
│   ├── multiple_distortion/    # Composite degradation generation and evaluation
│   ├── paper/                  # Manuscript source and experiment plan
│   └── config.py               # Dataset, checkpoint, and result paths
├── ablation/                   # Earlier controlled ablation experiments
├── re_dataset/                 # Dataset loaders
└── Restormer + Volterra/       # Earlier experimental workspace
```

---

## 한국어 소개

VTN-IR은 계층적 Transformer 인코더-디코더에 **절단된 2차 Volterra 연산자(truncated second-order Volterra operator)**를 결합한 통합 이미지 복원 네트워크입니다. 기존 Transformer에서 주의집중, 활성화 함수, 게이팅을 통해 암묵적으로 학습되던 특징 간 곱셈적 상호작용을 VET 블록 내부에서 명시적으로 모델링합니다.

하나의 공유 네트워크가 작업별 분기나 조건부 모듈 없이 다음 복원 작업을 처리합니다.

- 비 제거
- 모션 디블러링
- 안개 제거
- 눈 제거
- 통합 All-in-One 이미지 복원
- 복합 열화 이미지 복원

### 핵심 기여

- **명시적인 2차 상호작용 모델링:** 1차 합성곱 응답과 저랭크 2차 응답을 결합합니다.
- **VET 블록:** Volterra 연산자를 합성곱 주의집중 경로(MDTA)와 게이트 합성곱 피드포워드 경로(GDFN) 뒤에 모두 적용합니다.
- **통합 이미지 복원:** 열화 종류별 전용 분기 없이 하나의 공유 구조로 서로 다른 열화와 복합 열화를 처리합니다.
- **랭크 기반 복잡도 조절:** 완전한 2차 커널을 직접 만들지 않고 저랭크 분해를 사용하며, 기본 모델의 랭크는 \(R=4\)입니다.

### 실험 설정

VTN-IR은 다음 세 가지 조건에서 평가했습니다.

1. **단일 작업 복원:** 각 열화 종류에 대해 모델을 개별적으로 학습하고 평가합니다.
2. **통합 복원:** 비, 흐림, 안개, 눈 데이터를 혼합하여 하나의 모델을 학습합니다.
3. **복합 열화 복원:** Rain+Haze, Rain+Blur, Haze+Snow처럼 여러 열화가 동시에 존재하는 입력을 평가합니다.

평가 지표로 PSNR과 SSIM을 사용했습니다. 세부 정량 결과는 위의 [Results](#results) 표에서 확인할 수 있습니다.
