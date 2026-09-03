import type { Statistics } from "../types";

interface Props {
  stats: Statistics | null;
}

export default function StatsBar({ stats }: Props) {
  const items: { label: string; value: number | string }[] = [
    { label: "PERSONS", value: stats?.persons ?? "—" },
    { label: "VEHICLES", value: stats?.vehicles ?? "—" },
    { label: "FACES", value: stats?.faces_detected ?? "—" },
    { label: "IDENTIFIED", value: stats?.recognized_faces ?? "—" },
    { label: "ACTIVE ALERTS", value: stats?.active_events ?? "—" },
  ];

  return (
    <div className="stats-bar">
      {items.map((it) => (
        <div className="stat-card" key={it.label}>
          <div className="stat-value">{it.value}</div>
          <div className="stat-label">{it.label}</div>
        </div>
      ))}
    </div>
  );
}
