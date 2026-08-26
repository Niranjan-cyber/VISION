from typing import List, Dict, Any
import numpy as np

class ByteTrackerHandler:
    """ByteTrack wrapper for multi-object tracking and trajectory calculation."""

    def __init__(self):
        self.active_tracks = {}

    def update(self, detections: List[Dict[str, Any]], frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Updates tracks with new detections.
        Calculates bottom-center point for spatial position tracking: ( (x1+x2)//2, y2 ).
        """
        tracks = []
        for det in detections:
            bbox = det.get("bbox", [0, 0, 0, 0])
            x1, y1, x2, y2 = bbox
            bottom_center = ((x1 + x2) // 2, y2)
            
            tracks.append({
                "track_id": det.get("track_id", 0),
                "class_name": det.get("class_name", "unknown"),
                "bbox": bbox,
                "bottom_center": bottom_center,
            })
        return tracks
