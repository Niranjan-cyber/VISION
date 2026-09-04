import { Camera, Cpu, ShieldCheck } from "lucide-react";
import type { GlobalStatus } from "../types";

interface Props {
  status: GlobalStatus | null;
  connected: boolean;
  systemLabel: string;
  systemDotClass: string;
}

export default function SystemStatusBar({ status, connected, systemLabel, systemDotClass }: Props) {
  const cameraStatuses = status ? Object.values(status.cameras) : [];
  const onlineCount = cameraStatuses.filter((c) => c.status === "online").length;
  const anyOnline = onlineCount > 0;
  const cudaAccelerated = status?.ai_engine.yolo_device === "CUDA" || status?.ai_engine.face_recognition_device === "CUDA";

  return (
    <div className="panel system-status-bar">
      <div className="panel-header">
        <span>System Status</span>
      </div>
      <div className="system-status-body">
        <div className="system-status-headline">
          <span className={`dot ${systemDotClass}`} />
          {connected ? systemLabel : "CONNECTING TO BACKEND"}
        </div>
        <div className="system-status-facts">
          <span className="system-status-fact">
            <Camera size={14} strokeWidth={2} />
            {status ? `${onlineCount} Camera${onlineCount === 1 ? "" : "s"} Online` : "—"}
          </span>
          <span className="system-status-fact">
            <ShieldCheck size={14} strokeWidth={2} />
            AI Engine {anyOnline ? "Running" : "Idle"}
          </span>
          {status && (
            <span className="system-status-fact">
              <Cpu size={14} strokeWidth={2} />
              {cudaAccelerated ? "CUDA Accelerated" : "CPU Mode"}
            </span>
          )}
        </div>
      </div>
    </div>
  );
}
