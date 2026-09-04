# VISION Phase 3 — Operational Intelligence

Phase 1 built the AI pipeline (detection → tracking → face recognition →
events). Phase 2 made it a live multi-camera dashboard. Phase 3 turns the
events that pipeline already generates into something an operator can
actually **manage, investigate, search, and understand** — without touching
the AI pipeline, the multi-camera architecture, or the white/navy frontend
established in those earlier phases.

## Architecture

New module: `src/storage/` — a small SQLite-backed persistence layer,
deliberately not PostgreSQL/Redis/Kafka. It sits alongside the existing
`src/events/` package rather than replacing it:

```
src/events/            existing in-memory rule engine (SecurityEvent, Alert,
                        EventEngine) — unchanged, still the only place
                        events are *generated*
src/storage/
  db.py                 SQLite schema + one-connection-per-call access
  event_repository.py    EventRepository — persist/query SecurityEvents
  alert_repository.py    AlertRepository — persist/query Alerts, validates
                          the NEW → ACKNOWLEDGED → RESOLVED transition
  zone_repository.py      ZoneRepository — CRUD for Zones, replaces the
                          YAML file as the *live* source of zone geometry
  persistence_service.py  EventPersistenceService — glue: on a new event,
                          captures an annotated snapshot to disk and
                          writes both repositories
```

`CameraManager` owns one `Database` (and the repositories built on it) and
hands `zone_repo` / `persistence_service` to every `CameraSession` it
starts. A `CameraSession`'s AI worker thread — already decoupled from frame
capture/display (Phase 2.1) — calls `EventPersistenceService.record(...)`
right after `process_frame()` returns new events. This keeps persistence
off the live-video path entirely: writes only happen on a genuine new event
(not every frame), and any failure is caught and logged, never raised —
a database or disk problem cannot crash a camera's pipeline.

`PipelineSession` now **always** constructs an `EventEngine`, even with zero
zones (an empty zone list simply never fires a zone-based event — this is
what "NO ZONE CONFIGURED" already meant). This is what lets an operator add
a zone, later, to a camera that started with none: `zone_repo.list_for_camera()`
is the live source of truth; a YAML `zones_path` only ever *seeds* that
camera's row once (`ZoneRepository.seed_if_empty`), the first time it has no
rows — an operator's later edits are never clobbered by a restart. Editing
a zone via the API calls `CameraManager.refresh_camera_zones()`, which
re-reads from the repository and pushes the result into the running
session's `EventEngine.set_zones()` — no camera restart needed.

## Database schema

Three tables in one SQLite file (`data/vision.db` by default,
`VISION_DB_PATH` env var to override — tests always use an isolated temp
path so they never touch the real demo database):

```sql
events (
  id, camera_id, camera_name, source_type, event_type, severity,
  timestamp,       -- video-relative seconds (existing convention)
  created_at,      -- wall-clock ISO8601 — survives a restart
  track_id, identity, zone_id, zone_name, description, metadata,
  snapshot_path
)

alerts (
  id, event_id, camera_id, camera_name, severity, title, message,
  status,          -- NEW | ACKNOWLEDGED | RESOLVED
  created_at, acknowledged_at, resolved_at
)

zones (
  id, camera_id, name, type,   -- restricted | warning | monitored
  polygon,          -- JSON [[x,y], ...]
  enabled, created_at, updated_at
)
```

Indexed on `created_at`, `camera_id`, `event_type`, `severity`, `identity`,
`status` (alerts) — the columns Event History / Alerts filtering actually
queries on. No video/image blobs live in SQLite; a snapshot is a JPEG on
disk under `data/events/<camera_id>/<event_id>.jpg` (`VISION_SNAPSHOT_DIR`
to override), referenced by path in the `events` row.

## Event & alert lifecycle

```
EventEngine.update() fires a new (SecurityEvent, Alert) pair
        │
        ▼
CameraSession AI thread (already off the render path)
        │
        ▼
EventPersistenceService.record()
   ├── draw_annotations() on the frame just processed → JPEG → disk (best-effort)
   └── EventRepository.save() + AlertRepository.save()   (status = NEW)
```

Alert status only moves forward, enforced in one place
(`AlertRepository.transition`, `src/storage/alert_repository.py`):

```
NEW ──► ACKNOWLEDGED ──► RESOLVED
 └──────────────────────►
```

Any other transition (e.g. `RESOLVED → ACKNOWLEDGED`, `NEW → NEW`) raises
`InvalidAlertTransition`, surfaced by the API as `409 Conflict`. Acknowledging
or resolving also mutates the matching in-memory `Alert.status` on the live
camera's `EventEngine.active_alerts` (best-effort), so `GET
/cameras/{id}/events` — Phase 2's per-camera alert feed, left otherwise
unchanged — shows the same status as the new global `/alerts` endpoint.

## API endpoints

Unchanged: `/health`, `/status`, `/cameras*`, `/detections`,
`/cameras/{id}/*`. **Note:** `GET /events` changed meaning — in Phase 2 it
was the live alert feed; that role moved to `GET /alerts` (below), and
`GET /events` is now historical search. Not in the "never rename" list, and
the only consumer (the frontend) was updated in the same change.

```
GET  /alerts?camera_id=&severity=&status=&event_type=&start_time=&end_time=&limit=&offset=
GET  /alerts/{id}
POST /alerts/{id}/acknowledge      409 on an invalid transition
POST /alerts/{id}/resolve          409 on an invalid transition

GET  /events?camera_id=&event_type=&severity=&identity=&track_id=&status=&start_time=&end_time=&limit=&offset=
GET  /events/{id}
GET  /events/{id}/snapshot          404 if none was captured — never a fake image

GET  /investigations/event/{event_id}          event + its alert + related events (same camera+track)
GET  /investigations/person/{identity}          a *recognized* identity's events, across cameras
GET  /investigations/track/{camera_id}/{track_id}  an unrecognized person or any vehicle — the
                                                     only identifier the pipeline actually has for either

GET    /zones?camera_id=
GET    /zones/{id}
POST   /zones            {camera_id, name, type, polygon, enabled}
PUT    /zones/{id}       partial update (name/type/polygon/enabled)
DELETE /zones/{id}
```

`limit` is clamped server-side (max 500) on every list endpoint — a
malformed request can't pull the whole table.

## Zone management

The Zones page shows a camera's live feed with existing zones overlaid
(read from the same polygon data the event engine evaluates against — not a
separate drawing). Creating a zone is a click-drag rectangle: the browser
reads the `<img>`'s `naturalWidth/naturalHeight` (the frame's real
resolution) to convert the drag from displayed CSS pixels to the camera's
native pixel coordinates, so the stored polygon matches exactly what
`point_in_zone()` will test against. A zero-area polygon (a click with no
real drag) is rejected server-side, not just hidden in the UI. Disabling a
zone keeps it in the list (re-enable-able) but excludes it from the list
handed to the event engine — it stops generating events without being
deleted. A camera with no zones — recorded or live, uploaded or golden-demo
— shows "NO ZONE CONFIGURED" honestly; nothing is invented.

## Investigation workflow

```
Dashboard/Alerts → View → Investigation (event detail, real snapshot,
                                          incident timeline, related events)
                              │                    │
                        Acknowledge/Resolve   click a related event →
                                               re-investigate it
                              │
                    click Camera/Track/Identity →
                    Entity Investigation (person or track/vehicle)
```

The incident timeline is built only from data that exists: every event for
the same `(camera_id, track_id)` (oldest first), plus the alert's own
`acknowledged_at`/`resolved_at` if it has reached those states — never an
invented step. Person investigation aggregates a *recognized* identity's
events across every camera it was seen on; an unrecognized person or a
vehicle has no name, so it's investigated by `(camera_id, track_id)`
instead — the system never claims a cross-camera identity it didn't
establish. A missing ANPR plate reads "NOT AVAILABLE", never a fabricated
plate number.
