import type { GlobalStatus } from "../types";

interface Props {
  status: GlobalStatus | null;
  connected: boolean;
}

export default function Header({ status, connected }: Props) {
  const cameraStatuses = status ? Object.values(status.cameras) : [];
  const onlineCount = cameraStatuses.filter((c) => c.status === "online").length;
  const anyError = cameraStatuses.some((c) => c.status === "error");

  const online = connected && !!status && status.cameras_active > 0 && onlineCount === status.cameras_active;
  const label = !connected
    ? "CONNECTING..."
    : online
    ? "SYSTEM ONLINE"
    : anyError
    ? "SYSTEM DEGRADED"
    : "STARTING";
  const dotClass = !connected ? "dot-warn" : online ? "dot-ok" : anyError ? "dot-bad" : "dot-warn";

  return (
    <header className="app-header">
      <div>
        <div className="brand">VISION</div>
        <div className="tagline">MULTI-CAMERA BORDER SURVEILLANCE</div>
      </div>
      <div className="header-actions">
        {status && (
          <span className="cameras-online-tag mono">
            {onlineCount}/{status.cameras_active} cameras online
          </span>
        )}
        <div className="status-pill">
          <span className={`dot ${dotClass}`} />
          {label}
        </div>
      </div>
    </header>
  );
}
