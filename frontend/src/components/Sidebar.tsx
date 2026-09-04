import { Bell, LayoutDashboard, ListTree, MapPinned, ShieldCheck } from "lucide-react";

export type NavSection = "dashboard" | "alerts" | "events" | "zones";

interface Props {
  active: NavSection;
  onNavigate: (section: NavSection) => void;
  camerasOnline: number;
  camerasMax: number;
  activeAlertCount: number;
  systemLabel: string;
  systemDotClass: string;
}

const NAV_ITEMS: { id: NavSection; label: string; icon: typeof LayoutDashboard }[] = [
  { id: "dashboard", label: "Dashboard", icon: LayoutDashboard },
  { id: "alerts", label: "Alerts", icon: Bell },
  { id: "events", label: "Event History", icon: ListTree },
  { id: "zones", label: "Zones", icon: MapPinned },
];

export default function Sidebar({
  active,
  onNavigate,
  camerasOnline,
  camerasMax,
  activeAlertCount,
  systemLabel,
  systemDotClass,
}: Props) {
  return (
    <aside className="sidebar">
      <div className="sidebar-brand">
        <span className="sidebar-brand-mark">
          <ShieldCheck size={20} strokeWidth={2.25} />
        </span>
        <div>
          <div className="sidebar-brand-name">VISION</div>
          <div className="sidebar-brand-tagline">Intelligent Border Surveillance</div>
        </div>
      </div>

      <div className="sidebar-group-label">Monitor</div>
      <nav className="sidebar-nav">
        {NAV_ITEMS.map(({ id, label, icon: Icon }) => (
          <button
            key={id}
            className={`sidebar-nav-item ${active === id ? "active" : ""}`}
            onClick={() => onNavigate(id)}
          >
            <Icon size={16} strokeWidth={2} />
            <span>{label}</span>
            {id === "dashboard" && (
              <span className="sidebar-nav-count mono">
                {camerasOnline}/{camerasMax}
              </span>
            )}
            {id === "alerts" && activeAlertCount > 0 && (
              <span className="sidebar-nav-badge mono">{activeAlertCount}</span>
            )}
          </button>
        ))}
      </nav>

      <div className="sidebar-footer">
        <div className="sidebar-footer-status">
          <span className={`dot ${systemDotClass}`} />
          {systemLabel}
        </div>
      </div>
    </aside>
  );
}
