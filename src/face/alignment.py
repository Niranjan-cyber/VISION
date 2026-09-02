from typing import Optional, Tuple
import cv2
import numpy as np

# Standard 5-point reference facial landmarks for 112x112 ArcFace alignment
ARCFACE_REF_LANDMARKS = np.array(
    [
        [38.2946, 51.6963],  # right eye
        [73.5318, 51.5014],  # left eye
        [56.0252, 71.7366],  # nose tip
        [41.5493, 92.3655],  # right mouth corner
        [70.7299, 92.2041],  # left mouth corner
    ],
    dtype=np.float32,
)


def align_face(
    image: np.ndarray,
    landmarks: np.ndarray,
    target_size: Tuple[int, int] = (112, 112),
) -> Optional[np.ndarray]:
    """
    Applies 5-point facial landmark 2D similarity transform (affine alignment)
    to warp the source image into a 112x112 canonical ArcFace face crop.
    """
    if image is None or image.size == 0 or landmarks is None:
        return None

    src_pts = np.asarray(landmarks, dtype=np.float32)
    if src_pts.shape != (5, 2):
        return None

    if np.isnan(src_pts).any() or np.isinf(src_pts).any():
        return None

    try:
        M, _ = cv2.estimateAffinePartial2D(src_pts, ARCFACE_REF_LANDMARKS)
        if M is None:
            return None

        aligned = cv2.warpAffine(
            image, M, target_size, flags=cv2.INTER_LINEAR, borderValue=0
        )
        return aligned
    except Exception:
        return None
