from typing import Dict, Optional, Tuple, Union
import numpy as np

from src.core.types import FaceEmbedding, IdentityMatch
from src.face.gallery import FaceGallery
from src.face.preprocessing import l2_normalize


class FaceMatcher:
    """Performs cosine similarity search against an in-memory FaceGallery with threshold and margin enforcement."""

    TARGET_DIMENSION = 512

    def __init__(
        self,
        gallery: FaceGallery,
        threshold: float = 0.60,
        margin: float = 0.10,
    ):
        self.gallery = gallery
        self.threshold = threshold
        self.margin = margin

    def get_all_similarities(
        self, embedding: Union[FaceEmbedding, np.ndarray]
    ) -> Dict[str, float]:
        """
        Computes maximum cosine similarity for every enrolled identity in the gallery.
        Returns a dictionary mapping identity_name -> max_similarity_score.
        """
        if embedding is None:
            return {}

        if isinstance(embedding, FaceEmbedding):
            query_vec = embedding.vector
        elif isinstance(embedding, np.ndarray):
            query_vec = embedding
        else:
            return {}

        if query_vec is None or query_vec.size != self.TARGET_DIMENSION:
            return {}

        if np.isnan(query_vec).any() or np.isinf(query_vec).any():
            return {}

        if self.gallery is None or self.gallery.is_empty():
            return {}

        q_norm = l2_normalize(query_vec.astype(np.float32).flatten())
        results: Dict[str, float] = {}

        for identity in self.gallery.identities():
            ref_embeddings = self.gallery.get(identity)
            if not ref_embeddings:
                continue

            sims = [float(np.dot(q_norm, ref)) for ref in ref_embeddings]
            results[identity] = max(sims)

        return results

    def match(self, embedding: Union[FaceEmbedding, np.ndarray]) -> IdentityMatch:
        """
        Compares an incoming 512-dimensional face embedding against all enrolled reference vectors.
        Enforces both recognition_threshold and minimum_margin criteria:
            is_match = (best_similarity >= threshold) AND ((best_similarity - second_best_similarity) >= margin)
        Handles single-identity and empty galleries safely.
        """
        if embedding is None:
            return IdentityMatch(
                identity=None,
                similarity=0.0,
                is_match=False,
                second_similarity=0.0,
                margin=0.0,
            )

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

        if np.isnan(query_vec).any() or np.isinf(query_vec).any():
            return IdentityMatch(
                identity=None,
                similarity=0.0,
                is_match=False,
                second_similarity=0.0,
                margin=0.0,
            )

        if self.gallery is None or self.gallery.is_empty():
            return IdentityMatch(
                identity=None,
                similarity=0.0,
                is_match=False,
                second_similarity=0.0,
                margin=0.0,
            )

        all_sims = self.get_all_similarities(query_vec)
        if not all_sims:
            return IdentityMatch(
                identity=None,
                similarity=0.0,
                is_match=False,
                second_similarity=0.0,
                margin=0.0,
            )

        # Sort candidate scores in descending order
        sorted_candidates = sorted(
            all_sims.items(), key=lambda item: item[1], reverse=True
        )

        best_identity, best_similarity = sorted_candidates[0]

        if len(sorted_candidates) >= 2:
            second_best_identity, second_best_similarity = sorted_candidates[1]
            calc_margin = best_similarity - second_best_similarity
        else:
            second_best_similarity = 0.0
            calc_margin = best_similarity

        # Match criteria: threshold check AND margin check
        is_match = (best_similarity >= self.threshold) and (
            calc_margin >= self.margin
        )

        return IdentityMatch(
            identity=best_identity if is_match else None,
            similarity=best_similarity,
            is_match=is_match,
            second_similarity=second_best_similarity,
            margin=calc_margin,
        )
