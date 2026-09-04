import type { EventItem } from "../types";

interface Props {
  events: EventItem[];
  onSelectEvent?: (eventId: string) => void;
  title?: string;
  onViewAll?: () => void;
}

function formatEventType(eventType: string): string {
  return eventType.replace(/_/g, " ");
}

function formatWallClock(iso: string): string {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  return d.toLocaleTimeString("en-GB", { hour12: false });
}

/** Compact chronological event log — real historical events (GET /events),
 * not the operator alert feed (see GlobalAlerts). Reused both as the
 * Dashboard's "Recent Events" preview and inside the full Event History
 * page's results. */
export default function EventTimeline({ events, onSelectEvent, title = "Event Timeline", onViewAll }: Props) {
  return (
    <div className="panel timeline-panel">
      <div className="panel-header">
        <span>{title}</span>
        {onViewAll && (
          <button className="panel-header-link" onClick={onViewAll}>
            Event History
          </button>
        )}
      </div>
      {events.length === 0 ? (
        <div className="empty-state">No events recorded yet.</div>
      ) : (
        <div className="timeline-list">
          {events.map((e) => (
            <div
              className={`timeline-row ${onSelectEvent ? "timeline-row-clickable" : ""}`}
              key={e.event_id}
              onClick={() => onSelectEvent?.(e.event_id)}
            >
              <span className="mono timeline-time">{formatWallClock(e.created_at)}</span>
              <span className={`timeline-dot sev-dot-${e.severity.toLowerCase()}`} />
              <span className="timeline-type">{formatEventType(e.event_type)}</span>
              <span className="timeline-zone">{e.zone_name ?? "—"}</span>
              <span className="mono timeline-camera">{e.camera_id}</span>
              <span className="mono timeline-track">{e.track_id !== null ? `#${e.track_id}` : ""}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
