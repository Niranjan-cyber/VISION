import type { SystemStatus } from "../types";

interface Props {
  status: SystemStatus | null;
  connected: boolean;
}

const ROWS: { key: keyof SystemStatus; label: string }[] = [
  { key: "video", label: "VIDEO" },
  { key: "detection", label: "DETECTION" },
  { key: "tracking", label: "TRACKING" },
  { key: "face_id", label: "FACE ID" },
  { key: "anpr", label: "ANPR" },
  { key: "events", label: "EVENTS" },
];

export default function SystemStatusStrip({ status, connected }: Props) {
  return (
    <div className="status-strip">
      {ROWS.map(({ key, label }) => {
        const value = connected && status ? Boolean(status[key]) : false;
        return (
          <div className="status-item" key={key}>
            <span className={`dot ${value ? "dot-ok" : "dot-off"}`} />
            <span className="status-item-label">{label}</span>
            <span className={`status-item-value ${value ? "text-ok" : "text-off"}`}>
              {value ? "ONLINE" : "OFFLINE"}
            </span>
          </div>
        );
      })}
    </div>
  );
}
