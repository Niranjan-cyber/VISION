import VideoFeed from "./VideoFeed";
import { cameraStreamUrl } from "../api";
import type { AlertItem, CameraDetectionState } from "../types";

interface Props {
  camera: CameraDetectionState;
  alerts: AlertItem[]; // this camera's alerts only, already filtered by the caller
  onFocus: () => void;
}

const STATUS_LABEL: Record<string, string> = {
  starting: "STARTING",
  online: "ONLINE",
  stopped: "STOPPED",
  error: "ERROR",
};

export default function CameraCard({ camera, alerts, onFocus }: Props) {
  const highestSeverity = alerts.reduce<AlertItem["severity"] | null>((worst, a) => {
    const rank: Record<string, number> = { LOW: 1, MEDIUM: 2, HIGH: 3, CRITICAL: 4 };
    if (!worst || rank[a.severity] > rank[worst]) return a.severity;
    return worst;
  }, null);

  const activeZone = camera.persons.find((p) => p.zone)?.zone ?? camera.vehicles.find((v) => v.zone)?.zone;

  return (
    <button className={`camera-card status-${camera.camera_status}`} onClick={onFocus}>
      <div className="camera-card-head">
        <div>
          <div className="camera-card-name">{camera.camera_name}</div>
          <div className="camera-card-id mono">
            {camera.camera_id}
            <span className={`source-type-badge source-type-${camera.source_type}`}>
              {camera.source_type === "live" ? "● LIVE" : "RECORDED"}
            </span>
          </div>
        </div>
        <span className={`cam-status-pill cam-status-${camera.camera_status}`}>
          <span className="dot" />
          {STATUS_LABEL[camera.camera_status] ?? camera.camera_status.toUpperCase()}
        </span>
      </div>

      <div className="camera-card-video">
        {camera.camera_status === "online" && (
          <VideoFeed streamUrl={cameraStreamUrl(camera.camera_id)} alt={`Live feed — ${camera.camera_name}`} compact />
        )}
        {camera.camera_status === "starting" && <div className="cam-placeholder">Starting…</div>}
        {camera.camera_status === "error" && <div className="cam-placeholder cam-placeholder-error">Processing error</div>}
        {camera.camera_status === "stopped" && <div className="cam-placeholder">Stopped</div>}

        {highestSeverity && (
          <span className={`camera-alert-badge sev-${highestSeverity.toLowerCase()}`}>
            {alerts.length} ALERT{alerts.length > 1 ? "S" : ""}
          </span>
        )}
      </div>

      <div className="camera-card-foot">
        <span className={`cam-foot-item ${!camera.has_zone ? "cam-foot-no-zone" : ""}`}>
          {camera.has_zone ? activeZone ?? "No active zone" : "NO ZONE CONFIGURED"}
        </span>
        <span className="cam-foot-item mono">
          {camera.statistics.persons}p · {camera.statistics.vehicles}v
        </span>
      </div>
    </button>
  );
}
