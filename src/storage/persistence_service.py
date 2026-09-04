"""Bridges EventEngine output (SecurityEvent/Alert, in-memory, per-camera)
to the persistent store. Called from the AI worker thread — already off the
capture/render path (see src/pipeline/camera_manager.py) — right after
process_frame() returns new events, so persistence never touches frame
rendering. Every write is best-effort: a database or disk failure is logged
and swallowed, never raised, so one bad write can't take down a camera's
pipeline (Phase 3 spec's explicit error-handling requirement)."""
import os
import sys
from typing import List, Optional

import cv2
import numpy as np

from src.events.types import Alert, SecurityEvent
from src.storage.alert_repository import AlertRepository
from src.storage.event_repository import EventRepository

DEFAULT_SNAPSHOT_DIR = os.environ.get("VISION_SNAPSHOT_DIR", "data/events")
JPEG_QUALITY = 85


class EventPersistenceService:
    def __init__(
        self,
        event_repo: EventRepository,
        alert_repo: AlertRepository,
        snapshot_dir: str = DEFAULT_SNAPSHOT_DIR,
    ):
        self.event_repo = event_repo
        self.alert_repo = alert_repo
        self.snapshot_dir = snapshot_dir

    def record(
        self,
        camera_id: str,
        camera_name: str,
        source_type: str,
        new_events: List[SecurityEvent],
        new_alerts: List[Alert],
        annotated_frame: Optional[np.ndarray],
    ) -> None:
        # EventEngine.update() always appends events/alerts together in the
        # same order (one alert per event) — see src/events/engine.py.
        for evt, alr in zip(new_events, new_alerts):
            snapshot_path = None
            if annotated_frame is not None:
                snapshot_path = self._save_snapshot(camera_id, evt.event_id, annotated_frame)
            try:
                self.event_repo.save(
                    event_id=evt.event_id,
                    camera_id=camera_id,
                    camera_name=camera_name,
                    event_type=evt.event_type.value,
                    severity=evt.severity.value,
                    timestamp=evt.timestamp,
                    description=evt.message,
                    source_type=source_type,
                    track_id=evt.track_id,
                    identity=evt.metadata.get("identity"),
                    zone_id=evt.zone_id,
                    zone_name=evt.metadata.get("zone_name"),
                    metadata=evt.metadata,
                    snapshot_path=snapshot_path,
                )
                self.alert_repo.save(
                    alert_id=alr.alert_id,
                    event_id=alr.event_id,
                    camera_id=camera_id,
                    camera_name=camera_name,
                    severity=alr.severity.value,
                    title=alr.title,
                    message=alr.message,
                )
            except Exception as e:
                print(f"[WARNING] Failed to persist event {evt.event_id}: {e}", file=sys.stderr)

    def _save_snapshot(self, camera_id: str, event_id: str, frame: np.ndarray) -> Optional[str]:
        try:
            camera_dir = os.path.join(self.snapshot_dir, camera_id)
            os.makedirs(camera_dir, exist_ok=True)
            path = os.path.join(camera_dir, f"{event_id}.jpg")
            ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
            if not ok:
                return None
            with open(path, "wb") as f:
                f.write(buf.tobytes())
            return path
        except Exception as e:
            print(f"[WARNING] Failed to save event snapshot for {event_id}: {e}", file=sys.stderr)
            return None
