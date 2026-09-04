import { useEffect, useState } from "react";
import { ArrowLeft, Check, CheckCheck, ImageOff } from "lucide-react";
import { acknowledgeAlert, eventSnapshotUrl, fetchEventInvestigation, resolveAlert } from "../api";
import type { EventInvestigation } from "../types";

interface Props {
  eventId: string;
  onBack: () => void;
  onSelectEvent: (eventId: string) => void;
  onSelectTrack: (cameraId: string, trackId: number) => void;
  onSelectIdentity: (identity: string) => void;
}

function formatEventType(eventType: string): string {
  return eventType.replace(/_/g, " ");
}

function formatWallClock(iso: string): string {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  return d.toLocaleTimeString("en-GB", { hour12: false });
}

/** The core Phase 3 investigation screen — reached from an alert (View) or
 * an event history row. Only ever shows information the pipeline actually
 * produced: a missing snapshot says so plainly rather than showing a
 * placeholder image, an unrecognized face says NOT RECOGNIZED rather than
 * guessing an identity. */
export default function Investigation({ eventId, onBack, onSelectEvent, onSelectTrack, onSelectIdentity }: Props) {
  const [data, setData] = useState<EventInvestigation | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [snapshotFailed, setSnapshotFailed] = useState(false);

  async function load() {
    setError(null);
    setSnapshotFailed(false);
    try {
      setData(await fetchEventInvestigation(eventId));
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load investigation");
    }
  }

  useEffect(() => {
    load();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [eventId]);

  async function handleAcknowledge() {
    if (!data?.alert) return;
    setBusy(true);
    try {
      await acknowledgeAlert(data.alert.alert_id);
      await load();
    } finally {
      setBusy(false);
    }
  }

  async function handleResolve() {
    if (!data?.alert) return;
    setBusy(true);
    try {
      await resolveAlert(data.alert.alert_id);
      await load();
    } finally {
      setBusy(false);
    }
  }

  if (error) {
    return (
      <div className="panel">
        <div className="panel-header">
          <span>Incident Investigation</span>
        </div>
        <div className="empty-state">{error}</div>
      </div>
    );
  }
  if (!data) {
    return (
      <div className="panel">
        <div className="empty-state">Loading incident…</div>
      </div>
    );
  }

  const { event, alert, related_events } = data;

  // A compact, honest timeline: every related event (oldest first), this
  // event, then the alert's own real lifecycle timestamps if it has them —
  // never an invented step.
  type TimelineStep = { time: string; label: string; kind: "event" | "lifecycle" };
  const timelineSteps: TimelineStep[] = [
    ...related_events.map((e) => ({ time: e.created_at, label: formatEventType(e.event_type), kind: "event" as const })),
    { time: event.created_at, label: formatEventType(event.event_type), kind: "event" as const },
  ].sort((a, b) => a.time.localeCompare(b.time));
  if (alert?.acknowledged_at) {
    timelineSteps.push({ time: alert.acknowledged_at, label: "Alert acknowledged", kind: "lifecycle" as const });
  }
  if (alert?.resolved_at) {
    timelineSteps.push({ time: alert.resolved_at, label: "Alert resolved", kind: "lifecycle" as const });
  }
  timelineSteps.sort((a, b) => a.time.localeCompare(b.time));

  return (
    <div className="investigation">
      <div className="camera-detail-head">
        <button className="back-btn" onClick={onBack}>
          <ArrowLeft size={14} strokeWidth={2.25} />
          Back to Events
        </button>
        <div className="camera-detail-title">Incident Investigation</div>
      </div>

      <div className="main-grid">
        <div className="panel investigation-summary-panel">
          <div className="panel-header">
            <span>{formatEventType(event.event_type)}</span>
            <span className={`severity-pill severity-${event.severity.toLowerCase()}`}>{event.severity}</span>
          </div>
          <div className="investigation-facts">
            <div className="investigation-fact">
              <span className="investigation-fact-label">Camera</span>
              <button className="investigation-fact-link" onClick={() => onSelectTrack(event.camera_id, event.track_id ?? -1)}>
                {event.camera_id} — {event.camera_name}
              </button>
            </div>
            <div className="investigation-fact">
              <span className="investigation-fact-label">Time</span>
              <span className="mono">{formatWallClock(event.created_at)}</span>
            </div>
            <div className="investigation-fact">
              <span className="investigation-fact-label">Zone</span>
              <span>{event.zone_name ?? "NO ZONE CONFIGURED"}</span>
            </div>
            <div className="investigation-fact">
              <span className="investigation-fact-label">Track</span>
              {event.track_id !== null ? (
                <button className="investigation-fact-link mono" onClick={() => onSelectTrack(event.camera_id, event.track_id as number)}>
                  #{event.track_id}
                </button>
              ) : (
                <span>NOT AVAILABLE</span>
              )}
            </div>
            <div className="investigation-fact">
              <span className="investigation-fact-label">Identity</span>
              {event.identity && event.identity !== "UNKNOWN" ? (
                <button className="investigation-fact-link" onClick={() => onSelectIdentity(event.identity as string)}>
                  {event.identity}
                </button>
              ) : (
                <span>{event.identity === "UNKNOWN" ? "NOT RECOGNIZED" : "NO FACE DETECTED"}</span>
              )}
            </div>
            {alert && (
              <div className="investigation-fact">
                <span className="investigation-fact-label">Alert Status</span>
                <span className={`alert-status-badge alert-status-${alert.status.toLowerCase()}`}>{alert.status}</span>
              </div>
            )}
          </div>

          <div className="investigation-description">{event.description}</div>

          {alert && alert.status !== "RESOLVED" && (
            <div className="investigation-actions">
              {alert.status === "NEW" && (
                <button className="modal-cancel" disabled={busy} onClick={handleAcknowledge}>
                  <Check size={14} strokeWidth={2.25} />
                  Acknowledge
                </button>
              )}
              <button className="modal-submit" disabled={busy} onClick={handleResolve}>
                <CheckCheck size={14} strokeWidth={2.25} />
                Resolve
              </button>
            </div>
          )}
        </div>

        <div className="panel investigation-snapshot-panel">
          <div className="panel-header">
            <span>Event Frame</span>
          </div>
          {event.has_snapshot && !snapshotFailed ? (
            <img
              className="investigation-snapshot"
              src={eventSnapshotUrl(event.event_id)}
              alt={`Snapshot — ${formatEventType(event.event_type)}`}
              onError={() => setSnapshotFailed(true)}
            />
          ) : (
            <div className="investigation-snapshot-missing">
              <ImageOff size={28} strokeWidth={1.5} />
              <span>NO SNAPSHOT AVAILABLE</span>
            </div>
          )}
        </div>
      </div>

      <div className="main-grid">
        <div className="panel">
          <div className="panel-header">
            <span>Incident Timeline</span>
          </div>
          <div className="incident-timeline">
            {timelineSteps.map((step, i) => (
              <div className={`incident-timeline-step incident-timeline-${step.kind}`} key={`${step.time}-${i}`}>
                <div className="incident-timeline-marker">
                  <span className="incident-timeline-dot" />
                  {i < timelineSteps.length - 1 && <span className="incident-timeline-line" />}
                </div>
                <div className="incident-timeline-content">
                  <span className="mono incident-timeline-time">{formatWallClock(step.time)}</span>
                  <span className="incident-timeline-label">{step.label}</span>
                </div>
              </div>
            ))}
          </div>
        </div>

        <div className="panel">
          <div className="panel-header">
            <span>Related Events</span>
          </div>
          {related_events.length === 0 ? (
            <div className="empty-state">No other events for this track.</div>
          ) : (
            <EventTimelineRows events={related_events} onSelectEvent={onSelectEvent} />
          )}
        </div>
      </div>
    </div>
  );
}

function EventTimelineRows({ events, onSelectEvent }: { events: EventInvestigation["related_events"]; onSelectEvent: (id: string) => void }) {
  return (
    <div className="timeline-list">
      {events.map((e) => (
        <div className="timeline-row timeline-row-clickable" key={e.event_id} onClick={() => onSelectEvent(e.event_id)}>
          <span className="mono timeline-time">{formatWallClock(e.created_at)}</span>
          <span className={`timeline-dot sev-dot-${e.severity.toLowerCase()}`} />
          <span className="timeline-type">{formatEventType(e.event_type)}</span>
          <span className="mono timeline-camera">{e.camera_id}</span>
        </div>
      ))}
    </div>
  );
}
