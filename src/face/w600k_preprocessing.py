import numpy as np
import cv2
from typing import Optional


def preprocess_w600k_crop(
    face_crop: np.ndarray,
    target_size: tuple = (112, 112),
    is_bgr: bool = True,
) -> Optional[np.ndarray]:
    """
    Preprocesses face crop for the InsightFace w600k_r50.onnx recognition model.

    Model Contract:
    - Target Dimensions: 112x112 (H x W)
    - Color Format: BGR (native OpenCV layout) by default
    - Normalization: (pixel - 127.5) / 127.5 (scaled to [-1.0, 1.0])
    - Tensor Layout: NCHW [1, 3, 112, 112]
    - Data Type: float32
    """
    if face_crop is None or face_crop.size == 0:
        return None

    h, w = face_crop.shape[:2]
    if h < 4 or w < 4:
        return None

    # 1. Resize to target dimension (112, 112)
    resized = cv2.resize(face_crop, target_size, interpolation=cv2.INTER_LINEAR)

    # 2. Color channel formatting (InsightFace w600k expects BGR by default)
    if not is_bgr:
        img_fmt = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    else:
        img_fmt = resized

    # 3. Convert to float32 and normalize: (pixel - 127.5) / 127.5
    f_img = img_fmt.astype(np.float32)
    normalized = (f_img - 127.5) / 127.5

    # 4. Convert HWC -> CHW and add batch dimension -> NCHW [1, 3, 112, 112]
    chw = np.transpose(normalized, (2, 0, 1))
    tensor = np.ascontiguousarray(np.expand_dims(chw, axis=0), dtype=np.float32)

    return tensor
