# VISION — AI-Powered Border Surveillance & Video Analytics Platform

**VISION** is an AI-powered intelligent border video analytics platform: video ingestion, object detection (YOLO11n), multi-object tracking (ByteTrack), face detection/alignment (YuNet), face recognition (InsightFace W600K-R50), ANPR, and a deterministic event/alert engine (zones, intrusion, loitering, suspicious-vehicle detection) — now wired end-to-end into a live web dashboard.

This is an SIH hackathon MVP, not a production system. See **IMPLEMENTED NOW vs. FUTURE / PLANNED** below before assuming a capability exists.

---

## 📚 Project Documentation

- 📄 [docs/PROJECT_SUMMARY.md](docs/PROJECT_SUMMARY.md) — slice-by-slice history and benchmark metrics.
- 📄 [docs/VISION_PRD.md](docs/VISION_PRD.md) — the original PRD. **Describes a target architecture (SCRFD, LPRNet, etc.) that differs from what is actually implemented** (YuNet, custom ANPR) — treat it as a roadmap, not as documentation of current behavior.
- 📄 [docs/DASHBOARD_DESIGN_SPEC.md](docs/DASHBOARD_DESIGN_SPEC.md) — the dashboard/architecture/event-flow design specification for the next implementation pass (design-only, build target, not yet built).
- 📄 [configs/zones_demo.yaml](configs/zones_demo.yaml) — the calibrated golden demo zone, with notes on how it was derived.

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

Demo videos live in `data/videos/` and are **also gitignored** — they ship with the team's working copy but not with a fresh clone. The **official golden demo video is `data/videos/shreyas1.mp4`** (calibrated zone below assumes this exact clip). If you don't have it, any short clip of an enrolled person standing in frame will work, but you'll need to recalibrate `configs/zones_demo.yaml` for its resolution/framing (see the notes inside that file).

## 5. Face gallery setup

Enrolled identities live in `data/face_gallery/<Identity_Name>/*.jpg` (also gitignored — supply your own or copy the team's). The pipeline auto-enrolls every image found there on startup. With no gallery, every face will correctly show as **NO MATCH** rather than crash.

## 6. The golden demo zone

`configs/zones_demo.yaml` is calibrated specifically for `shreyas1.mp4` — the subject's actual bottom-center trajectory in that clip was measured (not guessed) before drawing the polygon. **`configs/zones.yaml`** (the original Slice 7 example) is generic and does **not** reliably fire events on any of the team's demo videos — use `zones_demo.yaml` for the live demo, not `zones.yaml`.

## 7. Backend startup

```bash
uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Runs the golden demo video through the real AI pipeline in a background thread (looping automatically at end-of-stream so the dashboard stays continuously live) and exposes it over HTTP. Configurable via env vars:

| Env var | Default | Purpose |
| :--- | :--- | :--- |
| `VISION_DEMO_VIDEO` | `data/videos/shreyas1.mp4` | Video the dashboard streams |
| `VISION_DEMO_ZONES` | `configs/zones_demo.yaml` | Zone config for event detection |
| `VISION_LOITERING_DURATION` | `3.0` | Seconds before `LOITERING` fires (short, because the golden clip is only ~4.3s of video) |
| `VISION_ENABLE_ANPR` | `false` | See ANPR note below — do not set `true` without first installing and validating `easyocr` |

Endpoints: `GET /health`, `GET /status`, `GET /detections`, `GET /events`, `GET /stream` (MJPEG), `POST /restart`.

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

On the golden demo video you should see: a person detected and tracked (persistent `track_id`), a face detected and recognized as **Shreyas_Chavan** (~71% similarity), an `INTRUSION` alert (HIGH) fire almost immediately once the person is in the marked zone, and a `LOITERING` alert (MEDIUM) fire ~3 seconds later. No vehicles/plates will appear — the golden clip has none, and ANPR is off by default (see below). The console should show **no** PostgreSQL connection errors unless you explicitly pass a `--db-uri`.

---

## ANPR decision (read before enabling)

ANPR is **disabled by default** for the live demo (`VISION_ENABLE_ANPR=false`, `--disable-anpr` on the CLI). Why: the default OCR path (`--ocr-engine auto`) silently falls back to a non-OCR heuristic stub that fabricates plate-shaped text (e.g. `"PLATE7"`) when `easyocr` isn't installed — and `easyocr` is **not** in `requirements.txt` or installed by default. No demo video currently has a verified-legible plate. **Do not enable ANPR for a live demo without first installing `easyocr` and manually confirming a real plate reads correctly on the specific video you'll show** — otherwise you risk displaying a fabricated plate number as if it were a real OCR result. The dashboard reflects the true state honestly: when ANPR is off, the vehicle panel says so and never shows placeholder plate data.

---

## IMPLEMENTED NOW

- Video ingestion, YOLO11n detection, ByteTrack tracking, YuNet face detection, InsightFace W600K-R50 recognition (Slices 1–5.7)
- ANPR pipeline (detector/enhancer/OCR/consensus) — implemented but **off by default** for the demo (see above)
- Event Intelligence & Alert Engine: `INTRUSION`, `UNKNOWN_PERSON_INTRUSION`, `LOITERING`, `SUSPICIOUS_VEHICLE`, zone geometry, deduplication (Slice 7)
- FastAPI backend (`backend/main.py`) serving the real pipeline's live state as JSON + MJPEG — no mock/parallel implementation
- React + Vite dashboard: live annotated video, persons/vehicles panels, security alerts panel, honest system-status indicators
- A calibrated, verified-working golden demo video/zone combination

## FUTURE / PLANNED

- Production authentication, multi-camera orchestration, cloud deployment, WebRTC, message queues/Kafka/Redis — intentionally **not** built; out of scope for this MVP
- PostgreSQL persistence exists (`src/face/vector_db.py`) but is optional and off by default; not required for the dashboard
- Adaptive/quality-aware face-recognition thresholds (documented as a known limitation in `docs/PROJECT_SUMMARY.md`, not implemented)
- Verified ANPR with a real OCR engine on a confirmed-legible plate
- Interactive alert Acknowledge/Resolve controls, multi-camera selection, historical event search — specified as future scope in [docs/DASHBOARD_DESIGN_SPEC.md](docs/DASHBOARD_DESIGN_SPEC.md#10-ui-element--backend-data-mapping); each would require a new backend endpoint that doesn't exist today

---

## 🧪 Test Suite Status

```bash
python -m unittest discover -s tests
```
142 tests pass (the original 129 covering the AI pipeline modules, plus 13 added for the pipeline-session refactor, the API serialization contract, and a live FastAPI smoke test — see `tests/test_pipeline_session.py`, `tests/test_pipeline_serialize.py`, `tests/test_backend_api.py`).
