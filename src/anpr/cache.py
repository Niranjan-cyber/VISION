from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

from src.core.types import BoundingBox, PlateRecognitionResult


@dataclass
class TrackPlateRecord:
    """Historical ANPR readings and aggregated consensus for a vehicle track."""

    track_id: int
    best_text: str
    best_confidence: float
    is_valid: bool
    latest_bbox: BoundingBox
    read_count: int = 1
    history: List[str] = field(default_factory=list)
    last_frame: int = 0


class PlateTrackCache:
    """
    Maintains temporal license plate consensus and caching across video frames.
    Prevents re-reading blur/occlusion on subsequent frames and locks in verified plates.
    """

    def __init__(self, confidence_boost_threshold: float = 0.05):
        self.confidence_boost_threshold = confidence_boost_threshold
        self._cache: Dict[int, TrackPlateRecord] = {}

    def update(
        self,
        track_id: int,
        plate: PlateRecognitionResult,
        frame_number: int,
    ) -> PlateRecognitionResult:
        """
        Updates the track's plate cache with a new observation and returns
        the highest-confidence consensus result.
        """
        if not plate.cleaned_text:
            if track_id in self._cache:
                rec = self._cache[track_id]
                return PlateRecognitionResult(
                    raw_text=rec.best_text,
                    cleaned_text=rec.best_text,
                    confidence=rec.best_confidence,
                    is_valid=rec.is_valid,
                    bbox=plate.bbox,
                )
            return plate

        if track_id not in self._cache:
            record = TrackPlateRecord(
                track_id=track_id,
                best_text=plate.cleaned_text,
                best_confidence=plate.confidence,
                is_valid=plate.is_valid,
                latest_bbox=plate.bbox,
                read_count=1,
                history=[plate.cleaned_text],
                last_frame=frame_number,
            )
            self._cache[track_id] = record
            return plate

        # Update existing track record
        rec = self._cache[track_id]
        rec.history.append(plate.cleaned_text)
        rec.read_count += 1
        rec.last_frame = frame_number
        rec.latest_bbox = plate.bbox

        # Decision rule for updating best reading:
        # 1. Full 9-10 char Indian plate ALWAYS beats any shorter fragment
        if len(plate.cleaned_text) in {9, 10} and len(rec.best_text) < 9:
            rec.best_text = plate.cleaned_text
            rec.best_confidence = plate.confidence
            rec.is_valid = True
        # 2. New plate is valid syntax while old is not
        elif plate.is_valid and not rec.is_valid:
            rec.best_text = plate.cleaned_text
            rec.best_confidence = plate.confidence
            rec.is_valid = True
        # 3. Both valid, but new is longer (prefer full 9-10 char plate over fragment)
        elif plate.is_valid and rec.is_valid and len(plate.cleaned_text) > len(rec.best_text):
            rec.best_text = plate.cleaned_text
            rec.best_confidence = plate.confidence
        # 4. Same validity and comparable length, but new has higher confidence
        elif (plate.is_valid == rec.is_valid) and (len(plate.cleaned_text) >= len(rec.best_text)) and (
            plate.confidence > (rec.best_confidence + self.confidence_boost_threshold)
        ):
            rec.best_text = plate.cleaned_text
            rec.best_confidence = plate.confidence

        # Multi-frame consensus voting:
        # If we have readings of full Indian plates (9-10 chars), ALWAYS prefer them over shorter fragments!
        full_indian = [p for p in rec.history if len(p) in {9, 10}]
        if full_indian:
            candidates_for_voting = full_indian
        else:
            candidates_for_voting = [p for p in rec.history if len(p) >= 7]

        if candidates_for_voting:
            len_counts = Counter(len(p) for p in candidates_for_voting)
            dominant_len, _ = len_counts.most_common(1)[0]
            same_len_plates = [p for p in candidates_for_voting if len(p) == dominant_len]
            if len(same_len_plates) >= 1:
                consensus_chars = []
                for idx in range(dominant_len):
                    char_col = [p[idx] for p in same_len_plates]
                    char_counts = Counter(char_col)
                    # Disambiguation for Indian plate registration digits (where 7 is thinned to 1)
                    if dominant_len in {9, 10} and idx >= (dominant_len - 4):
                        if "7" in char_counts and char_counts["7"] >= 1:
                            consensus_chars.append("7")
                            continue
                        if "3" in char_counts and char_counts["3"] >= 1 and char_counts.get("2", 0) <= char_counts["3"]:
                            consensus_chars.append("3")
                            continue
                    most_char = char_counts.most_common(1)[0][0]
                    consensus_chars.append(most_char)
                consensus_text = "".join(consensus_chars)
                rec.best_text = consensus_text
                rec.is_valid = True

        return PlateRecognitionResult(
            raw_text=plate.raw_text,
            cleaned_text=rec.best_text,
            confidence=rec.best_confidence,
            is_valid=rec.is_valid,
            bbox=plate.bbox,
        )

    def get(self, track_id: int) -> Optional[PlateRecognitionResult]:
        """Returns the current best recognized plate for a track ID, if available."""
        if track_id not in self._cache:
            return None
        rec = self._cache[track_id]
        return PlateRecognitionResult(
            raw_text=rec.best_text,
            cleaned_text=rec.best_text,
            confidence=rec.best_confidence,
            is_valid=rec.is_valid,
            bbox=rec.latest_bbox,
        )

    def prune_stale_tracks(self, active_track_ids: Set[int], max_idle_frames: int = 150) -> None:
        """Removes track cache entries that have not been observed for over max_idle_frames."""
        stale_ids = [
            tid for tid in self._cache
            if tid not in active_track_ids
        ]
        # Keep recent ones for persistence, prune if excess
        if len(self._cache) > 500:
            for tid in stale_ids[:100]:
                del self._cache[tid]

    def __len__(self) -> int:
        return len(self._cache)
