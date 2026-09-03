from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

from src.core.types import BoundingBox


@dataclass
class ObjectState:
    """
    Unified object state combining tracking, recognition, ANPR, and geometric context.
    Independent of detector/tracker implementation specifics.
    """

    track_id: int
    object_type: str
    bbox: BoundingBox
    confidence: float
    camera_id: str = "default"

    # Face recognition context
    identity: Optional[str] = None
    face_similarity: Optional[float] = None
    has_face_detected: bool = False

    # ANPR context
    plate: Optional[str] = None
    plate_confidence: Optional[float] = None

    # Temporal context
    first_seen: float = 0.0
    last_seen: float = 0.0

    # Zone context
    current_zone: Optional[str] = None
    previous_zone: Optional[str] = None

    # Spatial context (ground position: bottom-center of bounding box)
    position: Tuple[int, int] = field(default_factory=lambda: (0, 0))
    velocity: Tuple[float, float] = (0.0, 0.0)

    # Extensible metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        # Auto-compute bottom-center ground position if not explicitly provided
        if self.position == (0, 0) and self.bbox is not None:
            self.position = self.bbox.bottom_center
