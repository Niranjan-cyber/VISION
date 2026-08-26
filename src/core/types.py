from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class BoundingBox:
    """Bounding box coordinates in pixels (x1, y1, x2, y2)."""

    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self) -> int:
        return max(0, self.x2 - self.x1)

    @property
    def height(self) -> int:
        return max(0, self.y2 - self.y1)

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def center(self) -> Tuple[int, int]:
        return ((self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2)

    @property
    def bottom_center(self) -> Tuple[int, int]:
        return ((self.x1 + self.x2) // 2, self.y2)

    def as_tuple(self) -> Tuple[int, int, int, int]:
        return (self.x1, self.y1, self.x2, self.y2)


@dataclass
class Detection:
    """Represents a single object detection in VISION."""

    class_id: int
    class_name: str
    confidence: float
    bbox: BoundingBox
