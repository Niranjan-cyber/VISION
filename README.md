# VISION — AI-Powered Border Surveillance & Video Analytics Platform

**VISION** is an AI-powered intelligent border video analytics platform: video ingestion, object detection (YOLO11n), multi-object tracking (ByteTrack), face detection/alignment (YuNet), face recognition (InsightFace W600K-R50), ANPR, and a deterministic event/alert engine (zones, intrusion, loitering, suspicious-vehicle detection) — now wired end-to-end into a live web dashboard.

This is an SIH hackathon MVP, not a production system. See **IMPLEMENTED NOW vs. FUTURE / PLANNED** below before assuming a capability exists.

---

## 📚 Project Documentation

- 📄 [docs/PROJECT_SUMMARY.md](docs/PROJECT_SUMMARY.md) — slice-by-slice history and benchmark metrics.
- 📄 [docs/PHASE3.md](docs/PHASE3.md) — Phase 3 (Operational Intelligence): event/alert persistence, alert lifecycle, zone management, investigation workflow, database schema, new API endpoints.
- 📄 [docs/GPU_SETUP.md](docs/GPU_SETUP.md) — optional CUDA acceleration setup (YOLO + face recognition); the base install stays CPU-only/portable without it.
- 📄 [docs/VISION_PRD.md](docs/VISION_PRD.md) — the original PRD. **Describes a target architecture (SCRFD, LPRNet, etc.) that differs from what is actually implemented** (YuNet, custom ANPR) — treat it as a roadmap, not as documentation of current behavior.
- 📄 [docs/DASHBOARD_DESIGN_SPEC.md](docs/DASHBOARD_DESIGN_SPEC.md) — the original Phase 2 dashboard design spec (superseded in places by the white/navy redesign and Phase 3 — see docs/PHASE3.md for current UI structure).
- 📄 [configs/zones_demo.yaml](configs/zones_demo.yaml) — the calibrated golden demo zone. As of Phase 3, YAML zone files only *seed* a camera's zones the first time (see docs/PHASE3.md) — the live source of truth is the SQLite zone store, editable from the dashboard's Zones page.

---

## 1. Python & Node versions

- Python **3.11+** (developed/tested on 3.13). No other version is pinned in `requirements.txt`.
- Node **18+** (developed/tested on Node 24) for the React dashboard.

## 2. Install dependencies

```bash
python -m venv .venv
```
- Windows PowerShell: `.\.venv\Scripts\Activate.ps1`
- macOS/Linux: `source .venv/bin/activate`

```bash
pip install -r requirements.txt
pip install -r backend/requirements.txt
```

`requirements.txt` covers the AI pipeline (ultralytics, opencv-python, numpy, onnxruntime, psycopg2-binary). `backend/requirements.txt` adds `fastapi` + `uvicorn` for the dashboard API layer only — the AI pipeline itself has no FastAPI dependency.

## 3. Model setup

Model weights are **not committed to git** (`.gitignore` excludes `*.onnx`/`*.pt`) to keep the repo small. What's required and how each is obtained:

| Model | Path | How it's obtained |
| :--- | :--- | :--- |
| YOLO11n (detection) | `yolo11n.pt` (repo root) | Auto-downloaded by `ultralytics` on first run if missing. Needs internet the first time only. |
| YuNet (face detection) | `models/face_detection_yunet_2023mar.onnx` | Auto-downloaded from the OpenCV Zoo on first run if missing (see [src/face/detector.py](src/face/detector.py)). |
| InsightFace W600K-R50 (face recognition) | `models/w600k_r50.onnx` | Auto-downloaded from the official InsightFace `buffalo_l` release pack on first run if missing (~275MB one-time download — see [src/face/modern_embedder.py](src/face/modern_embedder.py)). |

If you already have these files locally (e.g. from a teammate), just drop them in the paths above and nothing will be downloaded.

## 4. Demo video assets

Demo videos live in `data/videos/` and are **also gitignored** — they ship with the team's working copy but not with a fresh clone. The default 4-camera dashboard needs four specific clips: `shreyas1.mp4`, `jaysingpure1.mp4`, `sample1.mp4`, `salman4.mp4` (see the camera table in §7 for which zone file pairs with each). If you're missing some of these, set `VISION_CAMERA_COUNT=1` to run just the original single-camera golden demo (`shreyas1.mp4`), or upload your own videos as additional cameras from the dashboard once it's running.

## 5. Face gallery setup

Enrolled identities live in `data/face_gallery/<Identity_Name>/*.jpg` (also gitignored — supply your own or copy the team's). The pipeline auto-enrolls every image found there on startup. With no gallery, every face will correctly show as **NO MATCH** rather than crash.

## 6. The golden demo zone

`configs/zones_demo.yaml` is calibrated specifically for `shreyas1.mp4` — the subject's actual bottom-center trajectory in that clip was measured (not guessed) before drawing the polygon. **`configs/zones.yaml`** (the original Slice 7 example) is generic and does **not** reliably fire events on any of the team's demo videos — use `zones_demo.yaml` for the live demo, not `zones.yaml`.

## 7. Backend startup (multi-camera)

```bash
uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Boots up to **4 simultaneous cameras**, each an independent `PipelineSession` (own video source, own tracks, own identities, own event history) running in its own thread via `CameraManager` (`src/pipeline/camera_manager.py`). Each camera's clip loops automatically at end-of-stream. The default 4-camera lineup:

| Camera | Video | Zones |
| :--- | :--- | :--- |
| CAM-01 "Border Gate" (the original golden demo) | `data/videos/shreyas1.mp4` | `configs/zones_demo.yaml` |
| CAM-02 "BOP East" | `data/videos/jaysingpure1.mp4` | `configs/zones_cam02.yaml` |
| CAM-03 "Perimeter Road" | `data/videos/sample1.mp4` | `configs/zones_cam03.yaml` |
| CAM-04 "Restricted Zone" | `data/videos/salman4.mp4` | `configs/zones_cam04.yaml` |

Configurable via env vars:

| Env var | Default | Purpose |
| :--- | :--- | :--- |
| `VISION_CAMERA_COUNT` | `4` | How many of the 4 default cameras to boot (`1` restores the original single-camera golden demo, and is what the test suite uses for a fast boot) |
| `VISION_CAM01_NAME` / `VISION_DEMO_VIDEO` / `VISION_DEMO_ZONES` | see above | Override CAM-01 specifically |
| `VISION_LOITERING_DURATION` | `3.0` | CAM-01's loitering duration (short, because that clip is only ~4.3s of video) |
| `VISION_ENABLE_ANPR` | `false` | Applies to every camera — see ANPR note below |

**Endpoints:**

| | |
| :--- | :--- |
| `GET /health` | liveness check |
| `GET /status` `GET /detections` `GET /events` | **global aggregate** across every active camera |
| `GET /cameras` | list all cameras + their summaries |
| `POST /cameras` | add a camera — multipart form, fields `camera_name` + `video` (file); rejects a 5th active camera with `409 Maximum 4 active camera streams reached.`; never trusts the original filename (stored under `data/uploads/<uuid>.<ext>`) |
| `DELETE /cameras/{id}` | remove a camera and free its slot |
| `POST /cameras/{id}/restart` | restart one camera (reopens its video, resets its own tracks/identities/events — every other camera is untouched) |
| `GET /cameras/{id}/status` `GET /cameras/{id}/detections` `GET /cameras/{id}/events` `GET /cameras/{id}/stream` | per-camera versions of the above |

A camera pointed at a bad/corrupt video reports `status: "error"` with a message and keeps the other 3 cameras running — one camera's failure never crashes the process or another camera (see `tests/test_camera_manager.py::TestCameraFailureIsolation`).

## 8. Frontend startup

```bash
cd frontend
npm install
npm run dev
```

Opens on `http://localhost:5173`, talks to the backend on `http://localhost:8000` (override with `VITE_API_BASE` if the backend runs elsewhere).

## 9. Demo commands

**Full dashboard (recommended for the live demo):** start the backend (§7) and frontend (§8), then open `http://localhost:5173`.

**CLI-only fallback** (if the dashboard isn't available — still live inference, just the original OpenCV window):
```bash
python -m src.main --video data/videos/shreyas1.mp4 --zones configs/zones_demo.yaml --loitering-duration 3 --disable-anpr
```
or simply run [`run_demo.ps1`](run_demo.ps1) on Windows, which runs the exact same command.

## 10. Expected behavior

On the dashboard's default 4-camera boot you should see all 4 tiles reach `LIVE` within ~20s, each with its own annotated stream. **CAM-01** and **CAM-02** each show one recognized identity (`Shreyas_Chavan` ~71%, `Atharva_Jaysingpure` ~69-78%) and fire `INTRUSION` + `LOITERING`. **CAM-03** and **CAM-04** show multiple simultaneous person tracks (unenrolled — expect `NOT RECOGNIZED`) and fire many more events, including `UNKNOWN_PERSON_INTRUSION`, since several different people cycle through their zones. No vehicles/plates will appear on any camera — ANPR is off by default (see below). The console should show **no** PostgreSQL connection errors unless you explicitly pass a `--db-uri`. Click any camera tile to open its focused view (large video, that camera's own alerts, its own person/vehicle tables); use **← All Cameras** to return to the grid.

---

## ANPR decision (read before enabling)

ANPR is **disabled by default** for the live demo (`VISION_ENABLE_ANPR=false`, `--disable-anpr` on the CLI). Why: the default OCR path (`--ocr-engine auto`) silently falls back to a non-OCR heuristic stub that fabricates plate-shaped text (e.g. `"PLATE7"`) when `easyocr` isn't installed — and `easyocr` is **not** in `requirements.txt` or installed by default. No demo video currently has a verified-legible plate. **Do not enable ANPR for a live demo without first installing `easyocr` and manually confirming a real plate reads correctly on the specific video you'll show** — otherwise you risk displaying a fabricated plate number as if it were a real OCR result. The dashboard reflects the true state honestly: when ANPR is off, the vehicle panel says so and never shows placeholder plate data.

---

## IMPLEMENTED NOW

- Video ingestion, YOLO11n detection, ByteTrack tracking, YuNet face detection, InsightFace W600K-R50 recognition (Slices 1–5.7)
- ANPR pipeline (detector/enhancer/OCR/consensus) — implemented but **off by default** for the demo (see above)
- Event Intelligence & Alert Engine: `INTRUSION`, `UNKNOWN_PERSON_INTRUSION`, `LOITERING`, `SUSPICIOUS_VEHICLE`, zone geometry, deduplication (Slice 7)
- **Multi-camera orchestration (Phase 2)**: up to 4 simultaneous, fully isolated `PipelineSession`s via `CameraManager` — recorded video *or* live local webcam/USB sources; add/remove/restart per camera; one camera's failure never affects another or crashes the process
- **Decoupled video streaming (Phase 2.1)**: frame capture/MJPEG display runs independently of AI inference (configurable `--ai-fps` / `VISION_AI_FPS`, default 8), so display never waits on inference; optional CUDA acceleration for YOLO + face recognition (`docs/GPU_SETUP.md`)
- **Operational Intelligence (Phase 3)** — see `docs/PHASE3.md` for full detail:
  - Persistent event/alert store (SQLite) — events and alerts survive a backend restart
  - Alert lifecycle: `NEW → ACKNOWLEDGED → RESOLVED`, enforced server-side, actionable from the dashboard
  - Historical Event Search with camera/type/severity/status filters
  - Incident Investigation: real captured event snapshot, incident timeline, related events (same camera+track)
  - Entity Investigation: a recognized person's events across cameras, or an unrecognized person/vehicle by (camera, track)
  - Zone Management from the dashboard: create/edit/delete/enable-disable zones by drawing on the live feed, backed by the same SQLite store (YAML zone files now only seed a camera's zones once)
- FastAPI backend (`backend/main.py`) serving per-camera and global-aggregate live state as JSON + per-camera MJPEG — no mock/parallel implementation
- React + Vite dashboard, redesigned as a white/navy SOC command center: sidebar navigation (Dashboard/Alerts/Event History/Zones), 4-camera grid (hero), click-to-focus single-camera view, camera management (add/remove/restart), honest per-camera status indicators
- A calibrated, verified-working golden 4-camera video/zone lineup (see §7)

## FUTURE / PLANNED

- Production authentication, cloud deployment, WebRTC, message queues/Kafka/Redis, Kubernetes/GPU orchestration — intentionally **not** built; out of scope for this MVP
- Real IP camera/RTSP ingestion — only local video files and local webcam/USB devices are supported; `VideoSource`/`PipelineSession` would need a new source abstraction for RTSP
- PostgreSQL persistence exists (`src/face/vector_db.py`, optional face-gallery storage) alongside Phase 3's SQLite event/alert/zone store — the two are unrelated and both off/on independently
- Adaptive/quality-aware face-recognition thresholds (documented as a known limitation in `docs/PROJECT_SUMMARY.md`, not implemented)
- Verified ANPR with a real OCR engine on a confirmed-legible plate
- True cross-camera person re-identification — a recognized identity's events are aggregated across cameras, but the system never claims to have proven the same *unrecognized* person moved between cameras (see `docs/PHASE3.md` §"Investigation workflow")

---

## 🧪 Test Suite Status

```bash
python -m unittest discover -s tests
```
172 tests pass (the original 129 covering the AI pipeline modules, 13 added for the single-camera pipeline-session refactor and API contract, and 30 more added for multi-camera orchestration — camera creation/removal, the 4-camera limit, cross-camera isolation, restart, upload validation, and failure isolation. See `tests/test_pipeline_session.py`, `tests/test_pipeline_serialize.py`, `tests/test_backend_api.py`, `tests/test_camera_manager.py`).
