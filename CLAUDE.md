# VISION — Project Notes for Claude

AI-powered border surveillance platform (SIH hackathon project). Multi-camera
dashboard: live detection/tracking, face recognition, zone-based event
detection (intrusion/loitering), alerting, and operational intelligence
(persistence, investigation, zone management).

## Critical: Python environment

**Always run the backend with the project's own virtualenv, never the system
or Anaconda Python:**

```
.\.venv\Scripts\python.exe -m uvicorn backend.main:app --reload
```

The `.venv` has CUDA-enabled `torch`/`onnxruntime-gpu` installed. Anaconda's
`python.exe` (or any other interpreter on PATH) only has CPU-only `torch`
and `onnxruntime` — launching with the wrong interpreter silently falls back
to CPU inference (device resolution is `auto`), which under multi-camera
concurrent load starves capture/display threads via GIL contention. This
looks like a video/rendering bug ("streams stuck on Connecting", "frames
take too long") but is actually a launch mistake, not a code regression.

If cameras seem slow or stuck, first check which interpreter launched the
running process before assuming a code bug:
```
wmic process where "ProcessId=<pid>" get CommandLine
```

## Architecture

- **Capture/AI decoupling**: video capture and display run independently of
  AI inference — a slow inference pass must never block the live video
  feed. Preserve this when touching `src/pipeline/`.
- **`src/pipeline/session.py`** (`PipelineSession`) — per-camera pipeline:
  detector, tracker, face recognition, event engine, zones. Always
  constructs an `EventEngine` (even with `zones=[]`) so zones can be added
  later to any camera via `_resolve_zones()` / `refresh_zones()`.
- **`src/pipeline/camera_manager.py`** — owns all camera sessions, the
  shared `Database`/repositories, and the `EventPersistenceService`. Each
  camera runs its own thread; camera failures must stay isolated (see
  `tests/test_camera_manager.py::TestCameraIsolation`).
- **`src/storage/`** — SQLite persistence (WAL mode, one connection per
  call, never shared across threads). Repository pattern:
  `EventRepository`, `AlertRepository`, `ZoneRepository`. DB path via
  `VISION_DB_PATH` env var (default `data/vision.db`).
- **`src/face/gallery.py`** — `load_gallery_from_dir_cached()` is a
  process-wide cache (keyed by absolute gallery dir + db_uri) so N cameras
  sharing one gallery directory don't each redo detect+align+embed for
  every gallery image. `FaceGallery` is never mutated after construction —
  safe to share read-only across camera threads. New identities just need a
  new subdirectory under `data/face_gallery/<Name>/`; no code changes.
- **Alert lifecycle**: `NEW → ACKNOWLEDGED → RESOLVED`, enforced in
  `AlertRepository.transition()`. Invalid transitions raise
  `InvalidAlertTransition` → HTTP 409.
- **API contract**: `GET /alerts` = live/persistent alert feed (lifecycle
  managed). `GET /events` = historical search (filters + pagination). These
  are deliberately different endpoints — don't conflate them.
- **Investigation model**: "related events" = same `(camera_id, track_id)`
  only. Person investigation aggregates by recognized identity name across
  cameras. Track investigation is per `(camera_id, track_id)` for
  unrecognized persons/vehicles. Never claim cross-camera re-identification
  — it isn't implemented.
- **Zones**: YAML-seed-once-then-SQLite-is-truth. `ZoneRepository.seed_if_empty()`
  only seeds a camera once; never clobbers operator edits made via the UI.

## Hard rules

- **Never fabricate data** — no fake GPS, plates, identities, timestamps,
  or relationships. If something can't be verified/reproduced, say so
  instead of inventing a plausible-looking value.
- **No Phase 4 features** unless explicitly asked (ANPR improvements,
  cross-camera re-ID, etc.) — stay in scope.
- **Preserve the white/navy SOC dashboard design system** in
  `frontend/` unless a redesign is explicitly requested.
- **Do not commit or push unless explicitly told to.** Default to leaving
  changes staged/unstaged for review.
- When a diagnosis-only bug report is requested, report root cause and the
  smallest safe fix — do not apply the fix until asked.

## Testing

Run the full suite before considering any change complete:
```
.\.venv\Scripts\python.exe -m pytest tests/ -v
```
269+ tests as of Phase 3 (storage, camera isolation, pipeline, backend API,
zones, alerts, investigations). If a test fails because behavior genuinely
changed on purpose, update the test and document why — don't hide it or
loosen the assertion.

## Docs

- [docs/PHASE3.md](docs/PHASE3.md) — persistence schema, alert lifecycle,
  zone management, investigation workflow, full endpoint list.
- [README.md](README.md) — implemented-vs-planned feature index.
