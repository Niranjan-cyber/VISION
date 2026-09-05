import os
import sys
import threading
from typing import Dict, List, Set, Union, Optional
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

    def __init__(self, db_uri: Optional[str] = None):
        self._gallery: Dict[str, List[np.ndarray]] = {}
        self._frames: Dict[str, List[np.ndarray]] = {}  # Store frame crops
        self.db = None
        if db_uri:
            from src.face.vector_db import PostgresVectorDatabase
            self.db = PostgresVectorDatabase(db_uri)

    def add(self, identity: str, embedding: Union[FaceEmbedding, np.ndarray], frame: Optional[np.ndarray] = None) -> None:
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
            self._frames[identity] = []

        self._gallery[identity].append(clean_copy)
        self._frames[identity].append(frame)

    def get(self, identity: str) -> List[np.ndarray]:
        """Returns all reference embeddings for the specified identity."""
        return self._gallery.get(identity, [])

    def get_frames(self, identity: str) -> List[np.ndarray]:
        """Returns all reference frames (face crops) for the specified identity."""
        return self._frames.get(identity, [])

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
    db_uri: Optional[str] = None,
) -> FaceGallery:
    """
    Populates a FaceGallery by scanning subdirectories in gallery_dir (e.g. data/face_gallery/<identity>/*),
    detecting faces, generating 512-d ArcFace embeddings, and enrolling valid embeddings.
    Also synchronizes with a persistent vector database if db_uri is provided.
    """
    from typing import Optional
    gallery = FaceGallery(db_uri=db_uri)

    # 1. Pre-load all existing enrolled users from the database if available
    if gallery.db is not None:
        try:
            print("[INFO] Pre-loading user embeddings and frames from PostgreSQL database...", file=sys.stderr)
            db_users = gallery.db.fetch_all_users()
            for identity, embedding, frame, source_name in db_users:
                gallery.add(identity, embedding, frame)
            print(f"[INFO] Loaded {len(db_users)} reference embeddings from database.", file=sys.stderr)
        except Exception as e:
            print(f"[WARNING] Could not pre-load users from database: {e}", file=sys.stderr)

    if not os.path.exists(gallery_dir):
        print(
            f"[INFO] Face gallery directory '{gallery_dir}' does not exist. Using database/empty gallery.",
            file=sys.stderr,
        )
        return gallery

    if not os.path.isdir(gallery_dir):
        print(
            f"[INFO] Face gallery path '{gallery_dir}' is not a directory. Using database/empty gallery.",
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
            f"[INFO] No identity subdirectories found in '{gallery_dir}'. Returning gallery (size: {len(gallery)}).",
            file=sys.stderr,
        )
        return gallery

    for identity in subdirs:
        identity_path = os.path.join(gallery_dir, identity)

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
            continue

        for img_path in image_files:
            filename = os.path.basename(img_path)
            # Use relative path as the unique source name key in database
            rel_path = os.path.relpath(img_path, gallery_dir)

            # Skip processing if already synced in database
            if gallery.db is not None:
                try:
                    if gallery.db.has_user_file(rel_path):
                        continue
                except Exception:
                    pass

            print(
                f"[INFO] Processing image '{filename}' for identity '{identity}'...",
                file=sys.stderr,
            )
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
                # Add in-memory
                gallery.add(identity, embedding, face_crop)
                print(
                    f"[INFO] Enrolled reference embedding for '{identity}' from '{filename}'.",
                    file=sys.stderr,
                )
                # Enroll in database
                if gallery.db is not None:
                    vec = embedding.vector if isinstance(embedding, FaceEmbedding) else embedding
                    gallery.db.add_user(identity, vec, face_crop, rel_path)
            else:
                print(
                    f"[WARNING] Failed to generate embedding for gallery image '{filename}'. Skipping.",
                    file=sys.stderr,
                )

    print("[INFO] Face gallery loading complete.", file=sys.stderr)
    print(f"[INFO] Total Identities: {len(gallery.identities())}", file=sys.stderr)
    print(f"[INFO] Total Embeddings: {len(gallery)}", file=sys.stderr)

    return gallery


_gallery_cache: Dict[str, FaceGallery] = {}
_gallery_cache_lock = threading.Lock()


def load_gallery_from_dir_cached(
    gallery_dir: str,
    face_detector,
    face_embedder,
    db_uri: Optional[str] = None,
) -> FaceGallery:
    """
    Process-wide cache around load_gallery_from_dir(), keyed by
    (gallery_dir, db_uri). Every camera needs its own face_detector/
    face_embedder for live per-frame processing, but the *reference*
    embeddings gallery loading produces are the same deterministic output
    no matter which detector/embedder instance computed them — so with N
    cameras sharing one gallery directory (the normal multi-camera setup),
    only the first caller actually does the detect+align+embed work for
    every gallery image; the rest get the already-built FaceGallery
    instantly instead of repeating it. A FaceGallery is never mutated after
    construction (see FaceGallery.add(), only called during loading), so
    sharing one instance read-only across camera threads is safe.
    """
    cache_key = f"{os.path.abspath(gallery_dir)}::{db_uri or ''}"
    with _gallery_cache_lock:
        cached = _gallery_cache.get(cache_key)
        if cached is not None:
            print(
                f"[INFO] Reusing already-loaded face gallery for '{gallery_dir}' "
                f"({len(cached)} embeddings, {len(cached.identities())} identities).",
                file=sys.stderr,
            )
            return cached
        gallery = load_gallery_from_dir(gallery_dir, face_detector, face_embedder, db_uri)
        _gallery_cache[cache_key] = gallery
        return gallery
