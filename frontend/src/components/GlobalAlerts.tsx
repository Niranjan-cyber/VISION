import { useState } from "react";
import { AlertOctagon, Check, CheckCheck, Eye } from "lucide-react";
import { acknowledgeAlert, resolveAlert } from "../api";
import type { AlertItem, Severity } from "../types";

interface Props {
  alerts: AlertItem[];
  onFocusCamera: (cameraId: string) => void;
  onInvestigate?: (eventId: string) => void;
  /** When set, only these statuses are shown — the compact Dashboard
   * preview passes ["NEW", "ACKNOWLEDGED"]; the full Alerts page omits this
   * to show everything, including RESOLVED. */
  statusFilter?: AlertItem["status"][];
  title?: string;
  onStatusChanged?: () => void;
  /** Caps how many cards render — the Dashboard's compact preview passes a
   * small number; the full Alerts page omits this to show everything. */
  limit?: number;
  onViewAll?: () => void;
}

const SEVERITY_META: Record<Severity, { className: string }> = {
  CRITICAL: { className: "sev-critical" },
  HIGH: { className: "sev-high" },
  MEDIUM: { className: "sev-medium" },
  LOW: { className: "sev-low" },
};

function formatTime(ts: number | null): string {
  if (ts === null) return "--:--";
  const m = Math.floor(ts / 60);
  const s = Math.floor(ts % 60);
  return `${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
}

function eventTypeFromTitle(title: string): string {
  return title.replace(/[^\x00-\x7F]/g, "").trim();
}

/** Judges should immediately see VISION is watching multiple locations —
 * every alert is tagged with which camera raised it, and clicking one jumps
 * straight to that camera's focus view. Phase 3: alerts now carry a real
 * NEW -> ACKNOWLEDGED -> RESOLVED lifecycle, actioned right from the card. */
export default function GlobalAlerts({
  alerts,
  onFocusCamera,
  onInvestigate,
  statusFilter,
  title = "Security Alerts",
  onStatusChanged,
  limit,
  onViewAll,
}: Props) {
  const [busy, setBusy] = useState<string | null>(null);
  const filtered = statusFilter ? alerts.filter((a) => statusFilter.includes(a.status)) : alerts;
  const visible = limit ? filtered.slice(0, limit) : filtered;

  async function handleAcknowledge(alertId: string, e: React.MouseEvent) {
    e.stopPropagation();
    setBusy(alertId);
    try {
      await acknowledgeAlert(alertId);
      onStatusChanged?.();
    } finally {
      setBusy(null);
    }
  }

  async function handleResolve(alertId: string, e: React.MouseEvent) {
    e.stopPropagation();
    setBusy(alertId);
    try {
      await resolveAlert(alertId);
      onStatusChanged?.();
    } finally {
      setBusy(null);
    }
  }

  return (
    <div className="panel alerts-panel">
      <div className="panel-header">
        <span>{title}</span>
        {onViewAll ? (
          <button className="panel-header-link" onClick={onViewAll}>
            View All ({filtered.length})
          </button>
        ) : (
          <span className="tag">{visible.length} shown</span>
        )}
      </div>
      <div className="alerts-list">
        {visible.length === 0 && <div className="empty-state">No alerts to show.</div>}
        {visible.map((a) => {
          const meta = SEVERITY_META[a.severity] ?? SEVERITY_META.MEDIUM;
          return (
            <div className={`alert-card ${meta.className}`} key={a.alert_id}>
              <button className="alert-card-body" onClick={() => onFocusCamera(a.camera_id)}>
                <div className="alert-top">
                  <span className="alert-severity">
                    <AlertOctagon size={13} strokeWidth={2.25} />
                    {a.severity}
                  </span>
                  <span className={`alert-status-badge alert-status-${a.status.toLowerCase()}`}>{a.status}</span>
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
              <div className="alert-actions">
                {a.status === "NEW" && (
                  <button
                    className="alert-action-btn"
                    disabled={busy === a.alert_id}
                    onClick={(e) => handleAcknowledge(a.alert_id, e)}
                  >
                    <Check size={13} strokeWidth={2.25} />
                    Acknowledge
                  </button>
                )}
                {a.status !== "RESOLVED" && (
                  <button
                    className="alert-action-btn"
                    disabled={busy === a.alert_id}
                    onClick={(e) => handleResolve(a.alert_id, e)}
                  >
                    <CheckCheck size={13} strokeWidth={2.25} />
                    Resolve
                  </button>
                )}
                {onInvestigate && (
                  <button
                    className="alert-action-btn alert-action-view"
                    onClick={(e) => {
                      e.stopPropagation();
                      onInvestigate(a.event_id);
                    }}
                  >
                    <Eye size={13} strokeWidth={2.25} />
                    View
                  </button>
                )}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
