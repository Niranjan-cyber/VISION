from typing import Union
import numpy as np

from src.core.types import FaceEmbedding, IdentityMatch
from src.face.gallery import FaceGallery
from src.face.preprocessing import l2_normalize


class FaceMatcher:
    """Performs cosine similarity search against an in-memory FaceGallery and enforces a recognition threshold."""

    TARGET_DIMENSION = 512

    def __init__(self, gallery: FaceGallery, threshold: float = 0.60):
        self.gallery = gallery
        self.threshold = threshold

    def match(self, embedding: Union[FaceEmbedding, np.ndarray]) -> IdentityMatch:
        """
        Compares an incoming 512-dimensional face embedding against all enrolled reference vectors in the gallery.
        Returns an IdentityMatch containing the best candidate identity, similarity score, and match boolean flag.
        """
        if embedding is None:
            return IdentityMatch(identity=None, similarity=0.0, is_match=False)

        if isinstance(embedding, FaceEmbedding):
            query_vec = embedding.vector
        elif isinstance(embedding, np.ndarray):
            query_vec = embedding
        else:
            raise TypeError(
                f"Expected FaceEmbedding or np.ndarray, got {type(embedding)}"
            )

        if query_vec is None or query_vec.size != self.TARGET_DIMENSION:
            raise ValueError(
                f"Query vector must be {self.TARGET_DIMENSION}-dimensional, got {query_vec.size if query_vec is not None else None}"
            )

        if self.gallery is None or self.gallery.is_empty():
            return IdentityMatch(identity=None, similarity=0.0, is_match=False)

        # Ensure query vector is L2-normalized float32 vector
        q_norm = l2_normalize(query_vec.astype(np.float32).flatten())

        best_identity = None
        best_similarity = -1.0

        for identity in self.gallery.identities():
            ref_embeddings = self.gallery.get(identity)
            if not ref_embeddings:
                continue

            # Calculate cosine similarity against all reference embeddings for this identity
            # Since vectors are L2-normalized, cosine similarity is the dot product.
            identity_sims = [
                float(np.dot(q_norm, ref)) for ref in ref_embeddings
            ]
            max_id_sim = max(identity_sims)

            if max_id_sim > best_similarity:
                best_similarity = max_id_sim
                best_identity = identity

        if best_similarity < 0.0:
            best_similarity = 0.0

        # Threshold check: similarity >= threshold -> Match
        is_match = best_similarity >= self.threshold

        if is_match:
            return IdentityMatch(
                identity=best_identity,
                similarity=best_similarity,
                is_match=True,
            )
        else:
            return IdentityMatch(
                identity=None,
                similarity=best_similarity,
                is_match=False,
            )
