import glob
import os
import sys
from typing import Dict, List, Union
import cv2
import numpy as np

from src.core.types import BoundingBox, FaceEmbedding
from src.face.preprocessing import l2_normalize


class FaceGallery:
    """In-memory gallery managing 512-dimensional L2-normalized reference face embeddings per identity."""

    TARGET_DIMENSION = 512

    def __init__(self):
        self._gallery: Dict[str, List[np.ndarray]] = {}

    def add(self, identity: str, embedding: Union[FaceEmbedding, np.ndarray]) -> None:
        """
        Enrolls a 512-dimensional face embedding for the given identity name.
        Ensures strict L2-normalization without mutating the input vector.
        """
        if not identity or not isinstance(identity, str):
            raise ValueError("Identity name must be a non-empty string")

        if isinstance(embedding, FaceEmbedding):
            vector = embedding.vector
        elif isinstance(embedding, np.ndarray):
            vector = embedding
        else:
            raise TypeError(
                f"Expected FaceEmbedding or np.ndarray, got {type(embedding)}"
            )

        if vector is None or vector.size != self.TARGET_DIMENSION:
            raise ValueError(
                f"Embedding vector must be {self.TARGET_DIMENSION}-dimensional, got {vector.size if vector is not None else None}"
            )

        flat_vec = vector.astype(np.float32).flatten()
        norm_vec = l2_normalize(flat_vec)

        clean_copy = np.copy(norm_vec)

        if identity not in self._gallery:
            self._gallery[identity] = []

        self._gallery[identity].append(clean_copy)

    def get(self, identity: str) -> List[np.ndarray]:
        """Returns all reference embeddings for the specified identity."""
        return self._gallery.get(identity, [])

    def identities(self) -> List[str]:
        """Returns a sorted list of all enrolled identity names."""
        return sorted(list(self._gallery.keys()))

    def is_empty(self) -> bool:
        """Returns True if the gallery contains no enrolled embeddings."""
        return len(self) == 0

    def __len__(self) -> int:
        """Returns total count of reference embeddings across all enrolled identities."""
        return sum(len(embeds) for embeds in self._gallery.values())


def load_gallery_from_dir(
    gallery_dir: str,
    face_detector,
    face_embedder,
) -> FaceGallery:
    """
    Populates a FaceGallery by scanning subdirectories in gallery_dir (e.g. data/face_gallery/<identity>/*.jpg),
    detecting faces, generating 512-d ArcFace embeddings, and enrolling valid embeddings.
    """
    gallery = FaceGallery()

    if not os.path.exists(gallery_dir):
        print(
            f"[INFO] Face gallery directory '{gallery_dir}' does not exist. Initializing empty gallery.",
            file=sys.stderr,
        )
        return gallery

    subdirs = [
        d
        for d in os.listdir(gallery_dir)
        if os.path.isdir(os.path.join(gallery_dir, d))
    ]

    if not subdirs:
        print(
            f"[INFO] No identity subdirectories found in '{gallery_dir}'. Initializing empty gallery.",
            file=sys.stderr,
        )
        return gallery

    valid_extensions = ("*.jpg", "*.jpeg", "*.png", "*.bmp")

    for identity in subdirs:
        identity_path = os.path.join(gallery_dir, identity)
        image_files = []
        for ext in valid_extensions:
            image_files.extend(glob.glob(os.path.join(identity_path, ext)))

        if not image_files:
            print(
                f"[WARNING] No image files found for identity '{identity}' in '{identity_path}'.",
                file=sys.stderr,
            )
            continue

        for img_path in image_files:
            img = cv2.imread(img_path)
            if img is None:
                print(
                    f"[WARNING] Unable to read image file '{img_path}'. Skipping.",
                    file=sys.stderr,
                )
                continue

            # Detect face in full image or crop
            faces = face_detector.detect(img)
            if not faces:
                print(
                    f"[WARNING] No face detected in gallery image '{img_path}'. Skipping.",
                    file=sys.stderr,
                )
                continue

            # Select highest confidence face in image
            best_face = max(faces, key=lambda f: f.confidence)
            fb = best_face.bbox
            fh, fw = img.shape[:2]

            fx1 = max(0, min(fb.x1, fw))
            fy1 = max(0, min(fb.y1, fh))
            fx2 = max(0, min(fb.x2, fw))
            fy2 = max(0, min(fb.y2, fh))

            if fx2 <= fx1 or fy2 <= fy1:
                print(
                    f"[WARNING] Invalid face box in gallery image '{img_path}'. Skipping.",
                    file=sys.stderr,
                )
                continue

            face_crop = img[fy1:fy2, fx1:fx2]
            embedding = face_embedder.embed(face_crop)

            if embedding is not None:
                gallery.add(identity, embedding)
                print(
                    f"[INFO] Enrolled reference embedding for '{identity}' from '{os.path.basename(img_path)}'."
                )
            else:
                print(
                    f"[WARNING] Failed to generate embedding for gallery image '{img_path}'. Skipping.",
                    file=sys.stderr,
                )

    print(
        f"[INFO] Face gallery loading complete. Total Identities: {len(gallery.identities())}, Total Embeddings: {len(gallery)}"
    )
    return gallery
