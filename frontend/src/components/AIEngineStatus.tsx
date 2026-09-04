import type { GlobalStatus } from "../types";

interface Props {
  status: GlobalStatus | null;
}

/** Compact, technical, supporting panel — never the dashboard's primary
 * element. Every row is a real device assignment from src/core/device.py,
 * reported through GET /status's ai_engine block; nothing here is invented. */
export default function AIEngineStatus({ status }: Props) {
  const engine = status?.ai_engine;
  const rows: { label: string; value: string }[] = engine
    ? [
        { label: "YOLO11n", value: engine.yolo_device },
        { label: "Face Recognition", value: engine.face_recognition_device },
        { label: "YuNet", value: engine.yunet_device },
        { label: "Tracking", value: engine.tracking_device },
        { label: "Event Engine", value: engine.event_engine_device },
      ]
    : [];

  return (
    <div className="panel ai-engine-panel">
      <div className="panel-header">
        <span>AI Engine</span>
        <span className={`ai-engine-running ${engine ? "" : "ai-engine-idle"}`}>
          <span className={`dot ${engine ? "dot-ok" : "dot-off"}`} />
          {engine ? "RUNNING" : "—"}
        </span>
      </div>
      {engine ? (
        <>
          <div className="ai-engine-rows">
            {rows.map((r) => (
              <div className="ai-engine-row" key={r.label}>
                <span className="ai-engine-label">{r.label}</span>
                <span className={`ai-engine-value ${r.value === "CUDA" ? "ai-engine-value-cuda" : ""}`}>
                  {r.value}
                </span>
              </div>
            ))}
          </div>
          <div className="ai-engine-fps">
            <span>AI FPS</span>
            <span className="mono">{engine.ai_fps.toFixed(1)}</span>
          </div>
        </>
      ) : (
        <div className="empty-state">Waiting for engine status…</div>
      )}
    </div>
  );
}
