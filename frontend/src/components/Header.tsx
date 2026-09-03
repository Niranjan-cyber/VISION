import { useState } from "react";
import { restartDemo } from "../api";
import type { SystemStatus } from "../types";

interface Props {
  status: SystemStatus | null;
  connected: boolean;
}

export default function Header({ status, connected }: Props) {
  const [restarting, setRestarting] = useState(false);

  // ANPR being off is a deliberate, honest state for this demo — not a
  // degraded system. Core perception subsystems must all be up for "online".
  const online =
    connected &&
    !!status &&
    status.video &&
    status.detection &&
    status.tracking &&
    status.face_id &&
    status.events;

  const label = !connected ? "CONNECTING..." : online ? "SYSTEM ONLINE" : "SYSTEM DEGRADED";
  const dotClass = !connected ? "dot-warn" : online ? "dot-ok" : "dot-warn";

  async function handleRestart() {
    setRestarting(true);
    try {
      await restartDemo();
    } finally {
      window.setTimeout(() => setRestarting(false), 1500);
    }
  }

  return (
    <header className="app-header">
      <div>
        <div className="brand">VISION</div>
        <div className="tagline">INTELLIGENT BORDER SURVEILLANCE</div>
      </div>
      <div className="header-actions">
        <button className="restart-btn" onClick={handleRestart} disabled={restarting || !connected}>
          {restarting ? "Restarting…" : "Restart Demo"}
        </button>
        <div className="status-pill">
          <span className={`dot ${dotClass}`} />
          {label}
        </div>
      </div>
    </header>
  );
}
