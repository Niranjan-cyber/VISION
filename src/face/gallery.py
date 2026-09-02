import os
import sys
from typing import Dict, List, Set, Union
import cv2
import numpy as np

from src.core.types import BoundingBox, FaceEmbedding
from src.face.alignment import align_face
from src.face.preprocessing import l2_normalize

SUPPORTED_IMAGE_EXTENSIONS: Set[str] = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
    ".jfif",
}


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
    Populates a FaceGallery by scanning subdirectories in gallery_dir (e.g. data/face_gallery/<identity>/*),
    detecting faces, generating 512-d ArcFace embeddings, and enrolling valid embeddings.
    """
    gallery = FaceGallery()

    if not os.path.exists(gallery_dir):
        print(
            f"[INFO] Face gallery directory '{gallery_dir}' does not exist. Initializing empty gallery.",
            file=sys.stderr,
        )
        return gallery

    if not os.path.isdir(gallery_dir):
        print(
            f"[INFO] Face gallery path '{gallery_dir}' is not a directory. Initializing empty gallery.",
            file=sys.stderr,
        )
        return gallery

    subdirs = sorted(
        [
            d
            for d in os.listdir(gallery_dir)
            if os.path.isdir(os.path.join(gallery_dir, d))
        ]
    )

    if not subdirs:
        print(
            f"[INFO] No identity subdirectories found in '{gallery_dir}'. Initializing empty gallery.",
            file=sys.stderr,
        )
        return gallery

    for identity in subdirs:
        identity_path = os.path.join(gallery_dir, identity)
        print(
            f"[INFO] Loading gallery identity '{identity}' from '{identity_path}'",
            file=sys.stderr,
        )

        all_entries = sorted(os.listdir(identity_path))
        image_files = []
        for entry in all_entries:
            entry_path = os.path.join(identity_path, entry)
            if not os.path.isfile(entry_path):
                continue
            ext = os.path.splitext(entry)[1].lower()
            if ext in SUPPORTED_IMAGE_EXTENSIONS:
                image_files.append(entry_path)

        if not image_files:
            print(
                f"[WARNING] No image files found for identity '{identity}' in '{identity_path}'.",
                file=sys.stderr,
            )
            continue

        print(
            f"[INFO] Found {len(image_files)} image file(s) for identity '{identity}'.",
            file=sys.stderr,
        )

        for img_path in image_files:
            filename = os.path.basename(img_path)
            img = cv2.imread(img_path)
            if img is None:
                print(
                    f"[WARNING] Unable to read gallery image '{filename}'. Skipping.",
                    file=sys.stderr,
                )
                continue

            faces = face_detector.detect(img)
            if not faces:
                print(
                    f"[WARNING] No face detected in gallery image '{filename}'. Skipping.",
                    file=sys.stderr,
                )
                continue

            best_face = max(faces, key=lambda f: f.confidence)
            
            # Apply 5-point facial landmark alignment if landmarks are available
            aligned_face = None
            if best_face.landmarks is not None:
                aligned_face = align_face(img, best_face.landmarks)

            if aligned_face is not None:
                face_crop = aligned_face
            else:
                fb = best_face.bbox
                fh, fw = img.shape[:2]

                fx1 = max(0, min(fb.x1, fw))
                fy1 = max(0, min(fb.y1, fh))
                fx2 = max(0, min(fb.x2, fw))
                fy2 = max(0, min(fb.y2, fh))

                if fx2 <= fx1 or fy2 <= fy1:
                    print(
                        f"[WARNING] Invalid face box in gallery image '{filename}'. Skipping.",
                        file=sys.stderr,
                    )
                    continue

                face_crop = img[fy1:fy2, fx1:fx2]

            embedding = face_embedder.embed(face_crop)

            if embedding is not None:
                gallery.add(identity, embedding)
                print(
                    f"[INFO] Enrolled reference embedding for '{identity}' from '{filename}'.",
                    file=sys.stderr,
                )
            else:
                print(
                    f"[WARNING] Failed to generate embedding for gallery image '{filename}'. Skipping.",
                    file=sys.stderr,
                )

    print("[INFO] Face gallery loading complete.", file=sys.stderr)
    print(f"[INFO] Total Identities: {len(gallery.identities())}", file=sys.stderr)
    print(f"[INFO] Total Embeddings: {len(gallery)}", file=sys.stderr)

    return gallery
