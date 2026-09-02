from typing import Optional, Tuple
import cv2
import numpy as np


def preprocess_face_crop(
    face_crop: np.ndarray,
    target_size: Tuple[int, int] = (112, 112),
) -> Optional[np.ndarray]:
    """
    Preprocesses a BGR face crop into the normalized blob format expected by ArcFace models.
    Converts BGR to RGB, scales pixel intensities to [-1, 1], and resizes to 112x112 (1, 3, 112, 112).
    Rejects invalid, empty, NaN, or Inf crops safely.
    """
    if face_crop is None or not isinstance(face_crop, np.ndarray) or face_crop.size == 0:
        return None

    if np.isnan(face_crop).any() or np.isinf(face_crop).any():
        return None

    h, w = face_crop.shape[:2]
    if h < 5 or w < 5:
        return None

    try:
        blob = cv2.dnn.blobFromImage(
            face_crop,
            scalefactor=1.0 / 127.5,
            size=target_size,
            mean=(127.5, 127.5, 127.5),
            swapRB=True,
            crop=False,
        )
        return blob
    except Exception:
        return None


def l2_normalize(vector: np.ndarray) -> np.ndarray:
    """
    Applies L2 normalization to a feature embedding vector.
    Rejects NaN or Inf inputs and safely handles zero-norm vectors without producing NaNs.
    Returns an independent float32 NumPy array.
    """
    if vector is None or not isinstance(vector, np.ndarray) or vector.size == 0:
        return np.zeros(512, dtype=np.float32)

    if np.isnan(vector).any() or np.isinf(vector).any():
        return np.zeros(512, dtype=np.float32)

    flat_vec = vector.astype(np.float32).flatten()
    norm = np.linalg.norm(flat_vec)

    if norm > 1e-12:
        norm_vec = flat_vec / norm
    else:
        norm_vec = np.zeros_like(flat_vec)

    # Final safety check against NaNs/Infs
    if np.isnan(norm_vec).any() or np.isinf(norm_vec).any():
        return np.zeros_like(flat_vec)

    return np.copy(norm_vec)
