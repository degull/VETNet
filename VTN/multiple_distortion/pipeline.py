# E:\restormer+volterra\Restormer + Volterra\multiple_distortion\pipeline.py

import random
import numpy as np
from .utils_data import distort_images   # ✅ ARNIQA 방식 distort_images 사용

def apply_random_distortions(img, Ndist: int = 4, num_levels: int = 5, return_info: bool = False):

    # Tensor로 변환
    import torchvision.transforms as T
    if not isinstance(img, np.ndarray) and not hasattr(img, "shape"):  # PIL.Image 인 경우
        img = T.ToTensor()(img)

    # Distortion 적용
    out, distort_functions, distort_values = distort_images(
        img, max_distortions=Ndist, num_levels=num_levels
    )

    if return_info:
        info = [f"{fn.__name__}(val={val})" for fn, val in zip(distort_functions, distort_values)]
        return out, info
    return out
