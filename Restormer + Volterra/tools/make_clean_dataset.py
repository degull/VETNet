import os
import shutil
import random

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

def list_images(root: str):
    files = []
    for dp, _, fns in os.walk(root):
        for fn in fns:
            if fn.lower().endswith(IMG_EXTS):
                files.append(os.path.join(dp, fn))
    return files

def safe_copy(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if not os.path.exists(dst):
        shutil.copy2(src, dst)

def main():
    # ✅ 너 경로에 맞춘 GT 소스들 (원하면 더 추가 가능)
    sources = [
        r"E:/restormer+volterra/data/rain100H/train/norain",
        r"E:/restormer+volterra/data/rain100L/train/norain",
        r"E:/restormer+volterra/data/RESIDE-6K/train/GT",
        r"E:/restormer+volterra/data/CSD/Train/Gt",
    ]

    OUT_ROOT = r"E:/restormer+volterra/data/CLEAN"
    OUT_TRAIN = os.path.join(OUT_ROOT, "train")
    OUT_TEST  = os.path.join(OUT_ROOT, "test")
    os.makedirs(OUT_TRAIN, exist_ok=True)
    os.makedirs(OUT_TEST, exist_ok=True)

    all_imgs = []
    for s in sources:
        if os.path.isdir(s):
            imgs = list_images(s)
            print(f"[Source] {s} -> {len(imgs)} images")
            all_imgs.extend(imgs)
        else:
            print(f"[Skip] Not found: {s}")

    if len(all_imgs) == 0:
        raise RuntimeError("No images collected. Check source paths.")

    # ✅ 섞고 train/test split (90/10)
    random.seed(1234)
    random.shuffle(all_imgs)

    n = len(all_imgs)
    n_test = max(1, int(n * 0.1))
    test_imgs = all_imgs[:n_test]
    train_imgs = all_imgs[n_test:]

    print(f"[Split] total={n} | train={len(train_imgs)} | test={len(test_imgs)}")

    # ✅ 파일명 충돌 방지: index 붙여서 저장
    for i, p in enumerate(train_imgs):
        ext = os.path.splitext(p)[1].lower()
        dst = os.path.join(OUT_TRAIN, f"clean_{i:07d}{ext}")
        safe_copy(p, dst)

    for i, p in enumerate(test_imgs):
        ext = os.path.splitext(p)[1].lower()
        dst = os.path.join(OUT_TEST, f"clean_{i:07d}{ext}")
        safe_copy(p, dst)

    print(f"[Done] CLEAN created at: {OUT_ROOT}")

if __name__ == "__main__":
    main()
