import { AlertOctagon } from "lucide-react";
import type { AlertItem, Severity } from "../types";

interface Props {
  alerts: AlertItem[];
  onFocusCamera: (cameraId: string) => void;
}

const SEVERITY_META: Record<Severity, { className: string }> = {
  CRITICAL: { className: "sev-critical" },
  HIGH: { className: "sev-high" },
  MEDIUM: { className: "sev-medium" },
  LOW: { className: "sev-low" },
};

function formatTime(ts: number): string {
  // ts is a video-relative timestamp (seconds since that clip started), not
  // a wall-clock time — render as mm:ss so it doesn't imply otherwise.
  const m = Math.floor(ts / 60);
  const s = Math.floor(ts % 60);
  return `${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
}

function eventTypeFromTitle(title: string): string {
  // Titles are human-readable (e.g. "🚨 INTRUSION DETECTED"); strip the emoji
  // for the compact badge.
  return title.replace(/[^\x00-\x7F]/g, "").trim();
}

/** Judges should immediately see VISION is watching multiple locations —
 * every alert is tagged with which camera raised it, and clicking one jumps
 * straight to that camera's focus view. */
export default function GlobalAlerts({ alerts, onFocusCamera }: Props) {
  return (
    <div className="panel alerts-panel">
      <div className="panel-header">
        <span>Security Alerts</span>
        <span className="tag">{alerts.length} active</span>
      </div>
      <div className="alerts-list">
        {alerts.length === 0 && <div className="empty-state">No active alerts.</div>}
        {alerts.map((a) => {
          const meta = SEVERITY_META[a.severity] ?? SEVERITY_META.MEDIUM;
          return (
            <button
              className={`alert-card ${meta.className}`}
              key={a.alert_id}
              onClick={() => onFocusCamera(a.camera_id)}
            >
              <div className="alert-top">
                <span className="alert-severity">
                  <AlertOctagon size={13} strokeWidth={2.25} />
                  {a.severity}
                </span>
                <span className="alert-camera-tag mono">{a.camera_id}</span>
                <span className="alert-time mono">{formatTime(a.timestamp)}</span>
              </div>
              <div className="alert-title">{eventTypeFromTitle(a.title)}</div>
              <div className="alert-camera-name">{a.camera_name}</div>
              {a.zone_name && <div className="alert-zone">{a.zone_name}</div>}
              {a.track_id !== null && (
                <div className="alert-meta mono">
                  Track #{String(a.track_id).padStart(2, "0")}
                  {a.object_type ? ` (${a.object_type})` : ""}
                </div>
              )}
            </button>
          );
        })}
      </div>
    </div>
  );
}
