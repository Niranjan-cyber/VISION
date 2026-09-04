import type { AlertItem } from "../types";

interface Props {
  alerts: AlertItem[]; // already newest-first from the backend
}

function formatTime(ts: number): string {
  const m = Math.floor(ts / 60);
  const s = Math.floor(ts % 60);
  return `${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
}

function eventTypeFromTitle(title: string): string {
  return title.replace(/[^\x00-\x7F]/g, "").trim();
}

/** Same underlying data as GlobalAlerts (GET /events) — a compact
 * chronological log across every camera, for scanning the whole sequence
 * rather than reacting to the newest alert. New component, zero backend
 * change (see docs/DASHBOARD_DESIGN_SPEC.md §1/§10). */
export default function EventTimeline({ alerts }: Props) {
  return (
    <div className="panel timeline-panel">
      <div className="panel-header">
        <span>Event Timeline</span>
      </div>
      {alerts.length === 0 ? (
        <div className="empty-state">No events recorded yet.</div>
      ) : (
        <div className="timeline-list">
          {alerts.map((a) => (
            <div className="timeline-row" key={a.alert_id}>
              <span className="mono timeline-time">{formatTime(a.timestamp)}</span>
              <span className={`timeline-dot sev-dot-${a.severity.toLowerCase()}`} />
              <span className="timeline-type">{eventTypeFromTitle(a.title)}</span>
              <span className="timeline-zone">{a.zone_name ?? "—"}</span>
              <span className="mono timeline-camera">{a.camera_id}</span>
              <span className="mono timeline-track">{a.track_id !== null ? `#${a.track_id}` : ""}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
