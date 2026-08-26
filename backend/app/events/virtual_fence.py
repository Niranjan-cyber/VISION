import cv2
import numpy as np
from typing import List, Tuple

class VirtualFence:
    """Polygon virtual fence intrusion detection using OpenCV pointPolygonTest()."""

    def __init__(self, zone_id: str, polygon_coords: List[Tuple[int, int]]):
        self.zone_id = zone_id
        self.polygon = np.array(polygon_coords, dtype=np.int32)

    def is_inside(self, point: Tuple[int, int]) -> bool:
        """
        Checks if point (bottom-center of bounding box) is inside the zone.
        pointPolygonTest returns >= 0 if inside or on edge.
        """
        res = cv2.pointPolygonTest(self.polygon, (float(point[0]), float(point[1])), measureDist=False)
        return res >= 0
