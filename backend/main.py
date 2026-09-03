"""
VISION FastAPI backend.

Thin API layer between the existing Python AI pipeline (src/pipeline/session.py)
and the React dashboard. This module contains NO detection/tracking/face/ANPR/
event logic of its own — it only:
  1. Runs one PipelineSession in a background thread against the golden demo
     video + zone config, looping at end-of-stream so the dashboard stays live.
  2. Publishes the session's latest state as JSON (GET /detections, /events)
     and its latest annotated frame as an MJPEG stream (GET /stream).

No Redis/Kafka/database — a single lock-guarded "latest state" snapshot is
enough for one demo camera feed, matching the existing project's scale.
"""
import os
import sys
import threading
import time
from contextlib import asynccontextmanager
from typing import Optional

# Ensure project root is in sys.path when running uvicorn from anywhere
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cv2
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from src.main import draw_annotations
from src.pipeline.serialize import serialize_events, serialize_state
from src.pipeline.session import PipelineSession, PipelineSubsystemError

# ---------------------------------------------------------------------------
# Golden demo configuration.
# See configs/zones_demo.yaml for why this specific video/zone pair was
# chosen and how the zone was calibrated. Override via env vars if needed.
# ---------------------------------------------------------------------------
VIDEO_PATH = os.environ.get("VISION_DEMO_VIDEO", "data/videos/shreyas1.mp4")
ZONES_PATH = os.environ.get("VISION_DEMO_ZONES", "configs/zones_demo.yaml")
LOITERING_DURATION = float(os.environ.get("VISION_LOITERING_DURATION", "3.0"))
STATIONARY_DURATION = float(os.environ.get("VISION_STATIONARY_DURATION", "60.0"))
# ANPR defaults OFF: no shipped demo video has a verified-legible plate, and
# EasyOCR is not installed — see docs/PROJECT_SUMMARY.md ANPR decision notes.
# Never silently fall back to the heuristic OCR stub for a live demo.
ENABLE_ANPR = os.environ.get("VISION_ENABLE_ANPR", "false").strip().lower() == "true"

JPEG_QUALITY = 80
STREAM_POLL_INTERVAL_S = 0.05

# ---------------------------------------------------------------------------
# Shared state between the background pipeline thread and the API handlers.
# ---------------------------------------------------------------------------
_lock = threading.Lock()
_session: Optional[PipelineSession] = None
_session_error: Optional[str] = None
_latest_result = None
_latest_jpeg: Optional[bytes] = None
_stop_flag = threading.Event()
_restart_flag = threading.Event()


def _build_session() -> PipelineSession:
    return PipelineSession(
        video_path=VIDEO_PATH,
        zones_path=ZONES_PATH,
        loitering_duration=LOITERING_DURATION,
        stationary_duration=STATIONARY_DURATION,
        enable_anpr=ENABLE_ANPR,
        verbose=True,
    )


def _pipeline_worker() -> None:
    global _session, _session_error, _latest_result, _latest_jpeg

    try:
        session = _build_session()
    except PipelineSubsystemError as e:
        with _lock:
            _session_error = f"{e.subsystem}: {e}"
        print(f"[BACKEND ERROR] Pipeline failed to start ({e.subsystem}): {e}", file=sys.stderr)
        return
    except Exception as e:
        with _lock:
            _session_error = str(e)
        print(f"[BACKEND ERROR] Pipeline failed to start: {e}", file=sys.stderr)
        return

    with _lock:
        _session = session

    print(f"[BACKEND] Pipeline session started. video={VIDEO_PATH} zones={ZONES_PATH} anpr={ENABLE_ANPR}", file=sys.stderr)

    while not _stop_flag.is_set():
        if _restart_flag.is_set():
            session.restart()
            _restart_flag.clear()
            print("[BACKEND] Session restarted for a fresh demo run.", file=sys.stderr)

        frame = session.source.read_frame()
        if frame is None:
            # Loop the golden clip so the dashboard stays continuously live.
            session.restart()
            continue

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

        with _lock:
            _latest_result = result
            if ok:
                _latest_jpeg = buf.tobytes()

    session.release()


_worker_thread: Optional[threading.Thread] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _worker_thread
    _worker_thread = threading.Thread(target=_pipeline_worker, daemon=True)
    _worker_thread.start()
    yield
    _stop_flag.set()


app = FastAPI(title="VISION Backend", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # MVP demo only, single trusted machine — see README
    allow_methods=["*"],
    allow_headers=["*"],
)


def _not_ready_response() -> JSONResponse:
    return JSONResponse(
        status_code=503,
        content={"error": _session_error or "pipeline is still starting"},
    )


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/status")
def status():
    with _lock:
        session = _session
        error = _session_error
    if session is None:
        return JSONResponse(
            status_code=503 if error else 200,
            content={
                "video": False,
                "detection": False,
                "tracking": False,
                "face_id": False,
                "anpr": False,
                "events": False,
                "error": error,
            },
        )
    return session.status


@app.get("/detections")
def detections():
    with _lock:
        session = _session
        result = _latest_result
    if session is None:
        return _not_ready_response()
    return serialize_state(session, result)


@app.get("/events")
def events():
    with _lock:
        session = _session
    if session is None:
        return _not_ready_response()
    return serialize_events(session)


@app.post("/restart")
def restart():
    with _lock:
        session = _session
    if session is None:
        return _not_ready_response()
    _restart_flag.set()
    return {"restarting": True}


def _mjpeg_generator():
    boundary = b"--frame"
    while True:
        with _lock:
            jpeg = _latest_jpeg
        if jpeg is not None:
            yield (
                boundary
                + b"\r\nContent-Type: image/jpeg\r\nContent-Length: "
                + str(len(jpeg)).encode()
                + b"\r\n\r\n"
                + jpeg
                + b"\r\n"
            )
        time.sleep(STREAM_POLL_INTERVAL_S)


@app.get("/stream")
def stream():
    return StreamingResponse(
        _mjpeg_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )
