"""
Multi-camera orchestration for VISION.

CameraManager owns 0..MAX_ACTIVE_CAMERAS independent CameraSession objects.
Each CameraSession wraps exactly one PipelineSession — the existing, unchanged
detection/tracking/face/ANPR/event pipeline. A camera can be a recorded video
file or a live local camera device; either way, the same PipelineSession
class runs it, never a parallel implementation.

Within one CameraSession, frame *acquisition* and *AI processing* run in two
independent threads sharing only a single "latest frame" / "latest AI
overlay" slot each (no queue, so latency can never grow unbounded):

    VideoCapture --> latest raw frame --> render+encode --> MJPEG (fast, every frame)
                  \\-> latest raw frame --> AI worker --> latest AI overlay (throttled to ai_fps)

The browser therefore always gets the newest frame immediately; the overlay
drawn on it is whatever the AI worker most recently finished (slightly
older), matching the golden demo's real-time feel without coupling display
FPS to inference FPS.

Failure isolation is the other core invariant this module provides: an
exception in one camera's threads must never propagate to another camera's
threads or to the FastAPI process itself.
"""
import sys
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set

import cv2

from src.core.types import FaceDetection, IdentityMatch, PlateRecognitionResult, Track
from src.face.association import FaceTrackAssociation
from src.main import draw_annotations
from src.pipeline.session import FrameResult, PipelineSession

MAX_ACTIVE_CAMERAS = 4
JPEG_QUALITY = 80
ERROR_POLL_INTERVAL_S = 0.3
AI_FPS_DEFAULT = 8.0
WORKER_JOIN_TIMEOUT_S = 5.0


class CameraStatus(str, Enum):
    """
    Four states, matching the design spec's ONLINE/PROCESSING/STOPPED/ERROR
    request with one deliberate merge: in this in-process, thread-per-camera
    design there is no moment where a session is initialized but not yet
    processing frames, so STARTING covers model/video load and ONLINE covers
    "actively processing" (the spec's ONLINE + PROCESSING collapse into one
    honest state rather than inventing a distinction that doesn't exist in
    the real implementation). A disconnected live camera or a processing
    exception both land in ERROR — `error` carries the specific reason
    (e.g. "camera disconnected") rather than a separate enum value.
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
    source_type: str = "video"  # "video" | "live"
    video_path: Optional[str] = None
    device_index: Optional[int] = None
    zones_path: Optional[str] = None
    loitering_duration: float = 3.0
    stationary_duration: float = 60.0
    movement_threshold: float = 15.0
    enable_anpr: bool = False
    device: str = "auto"
    ai_fps: float = AI_FPS_DEFAULT


@dataclass
class _OverlayState:
    """One coherent, immutable snapshot of the AI worker's latest output —
    everything the render step needs to draw an overlay, taken together so
    the capture thread never mixes tracks from one inference pass with
    identities from another."""

    tracks: List[Track] = field(default_factory=list)
    faces: List[FaceDetection] = field(default_factory=list)
    associations: List[FaceTrackAssociation] = field(default_factory=list)
    track_identity_map: Dict[int, IdentityMatch] = field(default_factory=dict)
    plates: List[PlateRecognitionResult] = field(default_factory=list)
    track_plate_map: Dict[int, PlateRecognitionResult] = field(default_factory=dict)
    breached_zone_ids: Set[str] = field(default_factory=set)


class CameraSession:
    """
    One camera: one PipelineSession, fully isolated per-camera state (its
    own lock, its own track IDs, its own event history), driven by two
    dedicated worker threads (capture, AI) plus a supervisor thread that
    owns init/restart/stop and failure isolation.
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
        self._latest_raw_frame = None
        self._latest_raw_frame_index = 0
        self._latest_overlay: Optional[_OverlayState] = None

        # Serializes anything that touches PipelineSession's cumulative AI
        # state (process_frame / restart) — the capture thread's EOF-loop
        # restart and the AI thread's process_frame() must never run
        # concurrently against the same tracker/gallery/event-engine state.
        self._pipeline_lock = threading.Lock()

        self._stop_flag = threading.Event()  # whole CameraSession teardown
        self._restart_flag = threading.Event()  # explicit user-requested restart
        self._workers_stop_flag = threading.Event()  # ask capture+AI threads to exit
        self._worker_error_flag = threading.Event()  # a worker hit an unrecoverable error

        self._capture_thread: Optional[threading.Thread] = None
        self._ai_thread: Optional[threading.Thread] = None

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
        self._thread.join(timeout=WORKER_JOIN_TIMEOUT_S)
        with self._lock:
            self._status = CameraStatus.STOPPED

    # ------------------------------------------------------------------
    # Internal — supervisor
    # ------------------------------------------------------------------
    def _set_status(self, status: CameraStatus, error: Optional[str] = None) -> None:
        with self._lock:
            self._status = status
            self._error = error

    def _try_init_session(self) -> Optional[PipelineSession]:
        cfg = self.config
        try:
            common = dict(
                zones_path=cfg.zones_path,
                loitering_duration=cfg.loitering_duration,
                stationary_duration=cfg.stationary_duration,
                movement_threshold=cfg.movement_threshold,
                enable_anpr=cfg.enable_anpr,
                camera_id=self.camera_id,
                device=cfg.device,
                verbose=False,
            )
            if cfg.source_type == "live":
                session = PipelineSession(device_index=cfg.device_index, **common)
            else:
                session = PipelineSession(video_path=cfg.video_path, **common)
        except Exception as e:
            self._set_status(CameraStatus.ERROR, f"failed to start: {e}")
            print(f"[CAMERA {self.camera_id}] failed to start: {e}", file=sys.stderr)
            return None

        with self._lock:
            self._session = session
        self._set_status(CameraStatus.ONLINE)
        return session

    def _wait_for_restart_or_stop(self) -> None:
        while not self._stop_flag.is_set() and not self._restart_flag.is_set():
            time.sleep(ERROR_POLL_INTERVAL_S)

    def _spawn_workers(self, session: PipelineSession) -> None:
        self._workers_stop_flag.clear()
        self._worker_error_flag.clear()
        with self._lock:
            self._latest_raw_frame = None
            self._latest_raw_frame_index = 0
            self._latest_overlay = None
        self._capture_thread = threading.Thread(
            target=self._capture_loop, args=(session,), daemon=True,
            name=f"camera-{self.camera_id}-capture",
        )
        self._ai_thread = threading.Thread(
            target=self._ai_loop, args=(session,), daemon=True,
            name=f"camera-{self.camera_id}-ai",
        )
        self._capture_thread.start()
        self._ai_thread.start()

    def _stop_workers(self) -> None:
        self._workers_stop_flag.set()
        if self._capture_thread is not None:
            self._capture_thread.join(timeout=WORKER_JOIN_TIMEOUT_S)
            self._capture_thread = None
        if self._ai_thread is not None:
            self._ai_thread.join(timeout=WORKER_JOIN_TIMEOUT_S)
            self._ai_thread = None

    def _run(self) -> None:
        session: Optional[PipelineSession] = None

        while session is None and not self._stop_flag.is_set():
            session = self._try_init_session()
            if session is None:
                self._wait_for_restart_or_stop()
                self._restart_flag.clear()
                # Loop back and retry init — picks up any corrected config
                # (e.g. a fixed video_path) on the next attempt.

        if session is None:
            return  # stopped before ever successfully starting

        self._spawn_workers(session)

        while not self._stop_flag.is_set():
            if self._restart_flag.is_set():
                self._restart_flag.clear()
                self._stop_workers()
                try:
                    with self._pipeline_lock:
                        session.restart()
                    self._set_status(CameraStatus.ONLINE)
                    self._spawn_workers(session)
                except Exception as e:
                    self._set_status(CameraStatus.ERROR, f"restart failed: {e}")
                    print(f"[CAMERA {self.camera_id}] restart failed: {e}", file=sys.stderr)
                    self._wait_for_restart_or_stop()
                continue

            if self._worker_error_flag.wait(timeout=ERROR_POLL_INTERVAL_S):
                # Failure isolation: a worker hit an unrecoverable error
                # (live disconnect, or an exception during process_frame)
                # and already set status=ERROR itself. This camera alone
                # idles here until an explicit restart/stop — nothing
                # crosses to another camera's threads or the FastAPI
                # process.
                self._stop_workers()
                self._worker_error_flag.clear()
                self._wait_for_restart_or_stop()

        self._stop_workers()
        try:
            session.release()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Internal — capture worker (fast: acquire + render + encode, no AI)
    # ------------------------------------------------------------------
    def _capture_loop(self, session: PipelineSession) -> None:
        is_live = session.is_live
        fps = session.source.fps or 30.0
        frame_interval = 1.0 / fps if fps > 0 else 1.0 / 30.0
        next_read_due = time.time()

        while not self._workers_stop_flag.is_set() and not self._stop_flag.is_set():
            frame = session.source.read_frame()

            if frame is None:
                if is_live:
                    # A live camera never loops — this is a genuine
                    # disconnect, surfaced honestly rather than retried
                    # silently. Only an explicit restart reopens the device.
                    self._set_status(CameraStatus.ERROR, "camera disconnected")
                    print(f"[CAMERA {self.camera_id}] live camera disconnected", file=sys.stderr)
                    self._workers_stop_flag.set()
                    self._worker_error_flag.set()
                    return

                # Recorded clip reached EOF -> loop it, same as the
                # single-threaded implementation always did. Reopening the
                # file and resetting tracker/gallery/event-engine state must
                # not overlap with the AI thread mid-process_frame().
                try:
                    with self._pipeline_lock:
                        session.restart()
                except Exception as e:
                    self._set_status(CameraStatus.ERROR, f"loop restart failed: {e}")
                    print(f"[CAMERA {self.camera_id}] loop restart failed: {e}", file=sys.stderr)
                    self._workers_stop_flag.set()
                    self._worker_error_flag.set()
                    return
                with self._lock:
                    self._latest_raw_frame = None
                    self._latest_raw_frame_index = 0
                    self._latest_overlay = None
                next_read_due = time.time()
                continue

            session.frame_index = session.source.current_frame
            with self._lock:
                self._latest_raw_frame = frame
                self._latest_raw_frame_index = session.frame_index
                overlay = self._latest_overlay

            annotated = draw_annotations(
                frame=frame,
                tracks=overlay.tracks if overlay else [],
                faces=overlay.faces if overlay else [],
                associations=overlay.associations if overlay else [],
                track_identity_map=overlay.track_identity_map if overlay else {},
                plates=overlay.plates if overlay else [],
                track_plate_map=overlay.track_plate_map if overlay else {},
                zones=session.zones,
                breached_zone_ids=overlay.breached_zone_ids if overlay else set(),
            )
            ok, buf = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])

            with self._lock:
                if ok:
                    self._latest_jpeg = buf.tobytes()
                if self._status != CameraStatus.ERROR:
                    self._status = CameraStatus.ONLINE

            if is_live:
                continue  # a live device's own capture call already paces itself

            # Recorded video: pace reads to the clip's native FPS so
            # playback looks realistic instead of decoding (and looping) as
            # fast as the CPU can read frames off disk.
            next_read_due += frame_interval
            sleep_for = next_read_due - time.time()
            if sleep_for > 0:
                time.sleep(sleep_for)
            else:
                next_read_due = time.time()  # fell behind — don't burst-catch-up

    # ------------------------------------------------------------------
    # Internal — AI worker (slow: full detection/tracking/face/ANPR/events)
    # ------------------------------------------------------------------
    def _ai_loop(self, session: PipelineSession) -> None:
        ai_fps = self.config.ai_fps if self.config.ai_fps and self.config.ai_fps > 0 else AI_FPS_DEFAULT
        ai_interval = 1.0 / ai_fps
        last_processed_index = -1

        while not self._workers_stop_flag.is_set() and not self._stop_flag.is_set():
            loop_start = time.time()

            with self._lock:
                frame = self._latest_raw_frame
                frame_index = self._latest_raw_frame_index

            if frame is not None and frame_index != last_processed_index:
                try:
                    with self._pipeline_lock:
                        result = session.process_frame(frame, frame_index)
                except Exception as e:
                    self._set_status(CameraStatus.ERROR, str(e))
                    print(f"[CAMERA {self.camera_id}] processing error: {e}", file=sys.stderr)
                    self._workers_stop_flag.set()
                    self._worker_error_flag.set()
                    return

                last_processed_index = frame_index
                overlay = _OverlayState(
                    tracks=session.latest_tracks,
                    faces=session.latest_faces,
                    associations=session.latest_associations,
                    track_identity_map=dict(session.track_identity_cache),
                    plates=session.latest_plates,
                    track_plate_map=dict(session.track_plate_map),
                    breached_zone_ids=set(session.latest_breached_zone_ids),
                )
                with self._lock:
                    self._latest_result = result
                    self._latest_overlay = overlay

            elapsed = time.time() - loop_start
            sleep_for = ai_interval - elapsed
            time.sleep(sleep_for if sleep_for > 0 else 0.01)


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
        video_path: Optional[str] = None,
        source_type: str = "video",
        device_index: Optional[int] = None,
        zones_path: Optional[str] = None,
        loitering_duration: float = 3.0,
        stationary_duration: float = 60.0,
        movement_threshold: float = 15.0,
        enable_anpr: bool = False,
        device: str = "auto",
        ai_fps: float = AI_FPS_DEFAULT,
    ) -> CameraSession:
        if source_type not in ("video", "live"):
            raise ValueError(f"source_type must be 'video' or 'live', got '{source_type}'")
        if source_type == "live" and device_index is None:
            raise ValueError("source_type='live' requires device_index")
        if source_type == "video" and not video_path:
            raise ValueError("source_type='video' requires video_path")

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
                source_type=source_type,
                video_path=video_path,
                device_index=device_index,
                zones_path=zones_path,
                loitering_duration=loitering_duration,
                stationary_duration=stationary_duration,
                movement_threshold=movement_threshold,
                enable_anpr=enable_anpr,
                device=device,
                ai_fps=ai_fps,
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
