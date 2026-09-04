import { useEffect, useState } from "react";
import type { GlobalStatus } from "../types";

interface Props {
  title: string;
  subtitle: string;
  status: GlobalStatus | null;
  connected: boolean;
  systemLabel: string;
  systemDotClass: string;
}

function useClock(): string {
  const [now, setNow] = useState(() => new Date());
  useEffect(() => {
    const id = window.setInterval(() => setNow(new Date()), 1000);
    return () => window.clearInterval(id);
  }, []);
  return now.toLocaleTimeString("en-GB", { hour12: false });
}

export default function TopBar({ title, subtitle, status, connected, systemLabel, systemDotClass }: Props) {
  const clock = useClock();
  const cameraStatuses = status ? Object.values(status.cameras) : [];
  const onlineCount = cameraStatuses.filter((c) => c.status === "online").length;

  return (
    <header className="topbar">
      <div>
        <div className="topbar-title">{title}</div>
        <div className="topbar-subtitle">{subtitle}</div>
      </div>
      <div className="topbar-actions">
        {status && (
          <span className="cameras-online-tag mono">
            {onlineCount}/{status.cameras_active} cameras online
          </span>
        )}
        <div className={`status-pill ${connected ? "" : "status-pill-muted"}`}>
          <span className={`dot ${systemDotClass}`} />
          {systemLabel}
        </div>
        <span className="topbar-clock mono">{clock}</span>
      </div>
    </header>
  );
}
