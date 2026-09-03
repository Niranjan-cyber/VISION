"""
Multi-camera orchestration for VISION.

CameraManager owns 0..MAX_ACTIVE_CAMERAS independent CameraSession objects.
Each CameraSession wraps exactly one PipelineSession — the existing, unchanged
detection/tracking/face/ANPR/event pipeline — running in its own background
thread. Four cameras therefore run four independent *instances* of the same
pipeline, never four different implementations of it.

Failure isolation is the core invariant this module exists to provide: an
exception in one camera's worker thread must never propagate to another
camera's thread or to the FastAPI process itself.
"""
import os
import sys
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional

import cv2

from src.main import draw_annotations
from src.pipeline.session import FrameResult, PipelineSession

MAX_ACTIVE_CAMERAS = 4
JPEG_QUALITY = 80
ERROR_POLL_INTERVAL_S = 0.3


class CameraStatus(str, Enum):
    """
    Four states, matching the design spec's ONLINE/PROCESSING/STOPPED/ERROR
    request with one deliberate merge: in this in-process, one-thread-per-
    camera design there is no moment where a session is initialized but not
    yet processing frames, so STARTING covers model/video load and ONLINE
    covers "actively processing" (the spec's ONLINE + PROCESSING collapse
    into one honest state rather than inventing a distinction that doesn't
    exist in the real implementation).
    """

    STARTING = "starting"
    ONLINE = "online"
    STOPPED = "stopped"
    ERROR = "error"


class CameraLimitReached(RuntimeError):
    """Raised when adding a camera would exceed MAX_ACTIVE_CAMERAS."""


@dataclass
class CameraConfig:
    camera_id: str
    camera_name: str
    video_path: str
    zones_path: Optional[str] = None
    loitering_duration: float = 3.0
    stationary_duration: float = 60.0
    movement_threshold: float = 15.0
    enable_anpr: bool = False


class CameraSession:
    """
    One camera: one PipelineSession, one dedicated worker thread, fully
    isolated per-camera state (its own lock, its own track IDs, its own
    event history — nothing here is shared with any other CameraSession).
    """

    def __init__(self, config: CameraConfig):
        self.config = config
        self.camera_id = config.camera_id
        self.camera_name = config.camera_name

        self._lock = threading.Lock()
        self._status = CameraStatus.STARTING
        self._error: Optional[str] = None
        self._session: Optional[PipelineSession] = None
        self._latest_jpeg: Optional[bytes] = None
        self._latest_result: Optional[FrameResult] = None

        self._stop_flag = threading.Event()
        self._restart_flag = threading.Event()
        self._thread = threading.Thread(
            target=self._run, daemon=True, name=f"camera-{self.camera_id}"
        )
        self._thread.start()

    # ------------------------------------------------------------------
    # Thread-safe accessors (called from FastAPI request handlers)
    # ------------------------------------------------------------------
    @property
    def status(self) -> str:
        with self._lock:
            return self._status.value

    @property
    def error(self) -> Optional[str]:
        with self._lock:
            return self._error

    def latest_jpeg(self) -> Optional[bytes]:
        with self._lock:
            return self._latest_jpeg

    def snapshot(self):
        """Returns (PipelineSession, latest FrameResult) as of one consistent instant."""
        with self._lock:
            return self._session, self._latest_result

    def request_restart(self) -> None:
        self._restart_flag.set()

    def stop(self) -> None:
        """Signals the worker thread to stop and blocks briefly for a clean shutdown."""
        self._stop_flag.set()
        self._thread.join(timeout=5.0)
        with self._lock:
            self._status = CameraStatus.STOPPED

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------
    def _set_status(self, status: CameraStatus, error: Optional[str] = None) -> None:
        with self._lock:
            self._status = status
            self._error = error

    def _try_init_session(self) -> Optional[PipelineSession]:
        try:
            session = PipelineSession(
                video_path=self.config.video_path,
                zones_path=self.config.zones_path,
                loitering_duration=self.config.loitering_duration,
                stationary_duration=self.config.stationary_duration,
                movement_threshold=self.config.movement_threshold,
                enable_anpr=self.config.enable_anpr,
                camera_id=self.camera_id,
                verbose=False,
            )
        except Exception as e:
            self._set_status(CameraStatus.ERROR, f"failed to start: {e}")
            print(f"[CAMERA {self.camera_id}] failed to start: {e}", file=sys.stderr)
            return None

        with self._lock:
            self._session = session
        self._set_status(CameraStatus.ONLINE)
        return session

    def _process_one_frame(self, session: PipelineSession) -> None:
        frame = session.source.read_frame()
        if frame is None:
            # End of clip — loop it so the tile stays alive for the demo.
            session.restart()
            return

        session.frame_index = session.source.current_frame
        result = session.process_frame(frame, session.frame_index)

        annotated = draw_annotations(
            frame=frame,
            tracks=session.latest_tracks,
            faces=session.latest_faces,
            associations=session.latest_associations,
            track_identity_map=session.track_identity_cache,
            plates=session.latest_plates,
            track_plate_map=session.track_plate_map,
            zones=session.zones,
            breached_zone_ids=session.latest_breached_zone_ids,
        )
        ok, buf = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])

        with self._lock:
            self._latest_result = result
            if ok:
                self._latest_jpeg = buf.tobytes()
            if self._status != CameraStatus.ERROR:
                self._status = CameraStatus.ONLINE

    def _wait_for_restart_or_stop(self) -> None:
        while not self._stop_flag.is_set() and not self._restart_flag.is_set():
            time.sleep(ERROR_POLL_INTERVAL_S)

    def _run(self) -> None:
        session: Optional[PipelineSession] = None

        while not self._stop_flag.is_set():
            if session is None:
                session = self._try_init_session()
                if session is None:
                    self._wait_for_restart_or_stop()
                    self._restart_flag.clear()
                    continue

            if self._restart_flag.is_set():
                try:
                    session.restart()
                    self._restart_flag.clear()
                    self._set_status(CameraStatus.ONLINE)
                except Exception as e:
                    self._restart_flag.clear()
                    self._set_status(CameraStatus.ERROR, f"restart failed: {e}")
                    print(f"[CAMERA {self.camera_id}] restart failed: {e}", file=sys.stderr)
                    self._wait_for_restart_or_stop()
                    continue

            try:
                self._process_one_frame(session)
            except Exception as e:
                # Failure isolation: this camera alone goes to ERROR and idles
                # until an explicit restart/stop — no exception crosses into
                # another camera's thread or the FastAPI process.
                self._set_status(CameraStatus.ERROR, str(e))
                print(f"[CAMERA {self.camera_id}] processing error: {e}", file=sys.stderr)
                self._wait_for_restart_or_stop()

        if session is not None:
            try:
                session.release()
            except Exception:
                pass


class CameraManager:
    """Owns every CameraSession. All mutation goes through this class so the
    active-camera count and camera_id allocation stay consistent."""

    def __init__(self):
        self._lock = threading.Lock()
        self._cameras: "OrderedDict[str, CameraSession]" = OrderedDict()
        self._next_camera_num = 1

    def add_camera(
        self,
        camera_name: str,
        video_path: str,
        zones_path: Optional[str] = None,
        loitering_duration: float = 3.0,
        stationary_duration: float = 60.0,
        movement_threshold: float = 15.0,
        enable_anpr: bool = False,
    ) -> CameraSession:
        with self._lock:
            if len(self._cameras) >= MAX_ACTIVE_CAMERAS:
                raise CameraLimitReached(
                    f"Maximum {MAX_ACTIVE_CAMERAS} active camera streams reached."
                )
            while True:
                camera_id = f"CAM-{self._next_camera_num:02d}"
                self._next_camera_num += 1
                if camera_id not in self._cameras:
                    break

            config = CameraConfig(
                camera_id=camera_id,
                camera_name=camera_name,
                video_path=video_path,
                zones_path=zones_path,
                loitering_duration=loitering_duration,
                stationary_duration=stationary_duration,
                movement_threshold=movement_threshold,
                enable_anpr=enable_anpr,
            )
            session = CameraSession(config)  # spawns its own thread; safe under this lock
            self._cameras[camera_id] = session
            return session

    def remove_camera(self, camera_id: str) -> bool:
        with self._lock:
            cam = self._cameras.pop(camera_id, None)
        if cam is None:
            return False
        cam.stop()
        return True

    def restart_camera(self, camera_id: str) -> bool:
        cam = self.get(camera_id)
        if cam is None:
            return False
        cam.request_restart()
        return True

    def get(self, camera_id: str) -> Optional[CameraSession]:
        with self._lock:
            return self._cameras.get(camera_id)

    def list_cameras(self) -> List[CameraSession]:
        with self._lock:
            return list(self._cameras.values())

    @property
    def active_count(self) -> int:
        with self._lock:
            return len(self._cameras)

    def shutdown(self) -> None:
        for cam in self.list_cameras():
            cam.stop()
