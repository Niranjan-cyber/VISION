import sys
from types import SimpleNamespace
from typing import Dict, List
import torch

from src.core.types import BoundingBox, Detection, Track

CLASS_NAME_MAP: Dict[int, str] = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}


class _DetectionResultsAdapter:
    """Adapter converting List[Detection] to the format expected by Ultralytics BYTETracker."""

    def __init__(self, detections: List[Detection]):
        if not detections:
            self.conf = torch.zeros((0,), dtype=torch.float32)
            self.cls = torch.zeros((0,), dtype=torch.float32)
            self.xywh = torch.zeros((0, 4), dtype=torch.float32)
            self.xyxy = torch.zeros((0, 4), dtype=torch.float32)
        else:
            conf_list = [d.confidence for d in detections]
            cls_list = [float(d.class_id) for d in detections]
            xyxy_list = [
                [
                    float(d.bbox.x1),
                    float(d.bbox.y1),
                    float(d.bbox.x2),
                    float(d.bbox.y2),
                ]
                for d in detections
            ]
            xywh_list = [
                [
                    (d.bbox.x1 + d.bbox.x2) / 2.0,
                    (d.bbox.y1 + d.bbox.y2) / 2.0,
                    float(d.bbox.width),
                    float(d.bbox.height),
                ]
                for d in detections
            ]

            self.conf = torch.tensor(conf_list, dtype=torch.float32)
            self.cls = torch.tensor(cls_list, dtype=torch.float32)
            self.xyxy = torch.tensor(xyxy_list, dtype=torch.float32)
            self.xywh = torch.tensor(xywh_list, dtype=torch.float32)

    def __getitem__(self, index):
        res = _DetectionResultsAdapter([])
        res.conf = self.conf[index]
        res.cls = self.cls[index]
        res.xyxy = self.xyxy[index]
        res.xywh = self.xywh[index]
        return res

    def __len__(self):
        return len(self.conf)


class ByteTrackTracker:
    """Multi-object tracker using ByteTrack for persistent ID association across frames."""

    def __init__(
        self,
        track_thresh: float = 0.25,
        match_thresh: float = 0.8,
        track_buffer: int = 30,
    ):
        self.track_thresh = track_thresh
        self.match_thresh = match_thresh
        self.track_buffer = track_buffer
        self.tracker = None
        self._init_tracker()

    def _init_tracker(self) -> None:
        """Initializes the underlying ByteTrack tracker instance."""
        try:
            from ultralytics.trackers.byte_tracker import BYTETracker
        except ImportError as e:
            print(
                "[ERROR] Failed to import BYTETracker from ultralytics. "
                "Ensure 'ultralytics' and 'lapx' packages are installed.",
                file=sys.stderr,
            )
            raise RuntimeError("BYTETracker dependency missing") from e

        args = SimpleNamespace(
            track_high_thresh=self.track_thresh,
            track_low_thresh=0.1,
            new_track_thresh=self.track_thresh,
            match_thresh=self.match_thresh,
            track_buffer=self.track_buffer,
            fuse_score=True,
        )

        try:
            self.tracker = BYTETracker(args)
        except Exception as e:
            print(f"[ERROR] ByteTrack tracker initialization failed: {e}", file=sys.stderr)
            raise RuntimeError(f"Unable to initialize ByteTrackTracker: {e}") from e

    def update(self, detections: List[Detection], frame_number: int) -> List[Track]:
        """
        Updates tracking state with frame detections and returns active Track objects.
        """
        if self.tracker is None:
            return []

        detection_cls_map = {d.class_id: d.class_name for d in detections}

        try:
            adapter = _DetectionResultsAdapter(detections)
            raw_tracks = self.tracker.update(adapter)
        except Exception as e:
            print(
                f"[WARNING] Tracker update failed at frame {frame_number}: {e}",
                file=sys.stderr,
            )
            return []

        tracks: List[Track] = []
        if raw_tracks is None or len(raw_tracks) == 0:
            return tracks

        for row in raw_tracks:
            x1, y1, x2, y2 = (
                int(round(row[0])),
                int(round(row[1])),
                int(round(row[2])),
                int(round(row[3])),
            )
            track_id = int(row[4])
            conf = float(row[5])
            cls_id = int(row[6])

            class_name = detection_cls_map.get(
                cls_id, CLASS_NAME_MAP.get(cls_id, f"class_{cls_id}")
            )

            bbox = BoundingBox(
                x1=max(0, x1),
                y1=max(0, y1),
                x2=max(0, x2),
                y2=max(0, y2),
            )

            track = Track(
                track_id=track_id,
                class_id=cls_id,
                class_name=class_name,
                confidence=conf,
                bbox=bbox,
                frame_number=frame_number,
            )
            tracks.append(track)

        return tracks

    def reset(self) -> None:
        """Resets tracker state."""
        self._init_tracker()
