import { useEffect, useState } from "react";
import Sidebar, { type NavSection } from "./components/Sidebar";
import TopBar from "./components/TopBar";
import SystemStatusBar from "./components/SystemStatusBar";
import AIEngineStatus from "./components/AIEngineStatus";
import CameraGrid from "./components/CameraGrid";
import CameraDetail from "./components/CameraDetail";
import GlobalAlerts from "./components/GlobalAlerts";
import GlobalStats from "./components/GlobalStats";
import EventTimeline from "./components/EventTimeline";
import EventHistory from "./components/EventHistory";
import Investigation from "./components/Investigation";
import EntityInvestigation, { type EntityQuery } from "./components/EntityInvestigation";
import ZoneManagement from "./components/ZoneManagement";
import CameraManagement from "./components/CameraManagement";
import AddCameraModal from "./components/AddCameraModal";
import {
  findCameraDetections,
  searchEvents,
  useCameraList,
  useGlobalAlerts,
  useGlobalDetections,
  useGlobalStatus,
} from "./api";
import type { EventItem, GlobalStatus } from "./types";

const MAX_SLOTS = 4;

const SECTION_META: Record<NavSection, { title: string; subtitle: string }> = {
  dashboard: { title: "Dashboard", subtitle: "Live surveillance feeds from connected border locations" },
  alerts: { title: "Alerts", subtitle: "AI-generated security events across the surveillance network" },
  events: { title: "Event History", subtitle: "Search and review the full historical event log" },
  zones: { title: "Zone Management", subtitle: "Inspect, create, and edit surveillance zones per camera" },
};

function systemStatus(status: GlobalStatus | null, connected: boolean) {
  const cameraStatuses = status ? Object.values(status.cameras) : [];
  const onlineCount = cameraStatuses.filter((c) => c.status === "online").length;
  const anyError = cameraStatuses.some((c) => c.status === "error");
  const online = connected && !!status && status.cameras_active > 0 && onlineCount === status.cameras_active;

  const label = !connected
    ? "CONNECTING"
    : online
    ? "ALL SYSTEMS OPERATIONAL"
    : anyError
    ? "SYSTEM DEGRADED"
    : "STARTING";
  const dotClass = !connected ? "dot-warn" : online ? "dot-ok" : anyError ? "dot-bad" : "dot-warn";
  return { label, dotClass };
}

/** Recent historical events for the Dashboard's compact preview widget —
 * a light on-demand poll, separate from the full Event History page's
 * filtered on-demand search. */
function useRecentEvents(intervalMs = 5000) {
  const [events, setEvents] = useState<EventItem[]>([]);
  useEffect(() => {
    let cancelled = false;
    async function tick() {
      try {
        const result = await searchEvents({ limit: 5 });
        if (!cancelled) setEvents(result);
      } catch {
        // keep showing the last good list
      }
    }
    tick();
    const id = window.setInterval(tick, intervalMs);
    return () => {
      cancelled = true;
      window.clearInterval(id);
    };
  }, [intervalMs]);
  return events;
}

type ViewMode =
  | { kind: "section" }
  | { kind: "camera"; cameraId: string }
  | { kind: "investigation"; eventId: string }
  | { kind: "entity"; query: EntityQuery };

export default function App() {
  const { data: detections, connected: detectionsConnected } = useGlobalDetections();
  const { data: alerts, connected: alertsConnected } = useGlobalAlerts();
  const { data: status, connected: statusConnected } = useGlobalStatus();
  const { data: cameras } = useCameraList();
  const recentEvents = useRecentEvents();

  const [activeSection, setActiveSection] = useState<NavSection>("dashboard");
  const [view, setView] = useState<ViewMode>({ kind: "section" });
  const [showAddModal, setShowAddModal] = useState(false);

  const cameraList = detections?.cameras ?? [];
  const alertList = alerts ?? [];
  const focusedCamera = view.kind === "camera" ? findCameraDetections(detections, view.cameraId) : null;
  const { label: systemLabel, dotClass: systemDotClass } = systemStatus(status, statusConnected);
  const onlineCount = status ? Object.values(status.cameras).filter((c) => c.status === "online").length : 0;

  function focusCamera(cameraId: string) {
    setView({ kind: "camera", cameraId });
  }
  function openInvestigation(eventId: string) {
    setView({ kind: "investigation", eventId });
  }
  function openEntity(query: EntityQuery) {
    setView({ kind: "entity", query });
  }
  function navigate(section: NavSection) {
    setActiveSection(section);
    setView({ kind: "section" });
  }
  function backToSection() {
    setView({ kind: "section" });
  }

  let title = SECTION_META[activeSection].title;
  let subtitle = SECTION_META[activeSection].subtitle;
  if (view.kind === "camera" && focusedCamera) {
    title = focusedCamera.camera_name;
    subtitle = focusedCamera.camera_id;
  } else if (view.kind === "investigation") {
    title = "Investigation";
    subtitle = "Incident detail, snapshot, and timeline";
  } else if (view.kind === "entity") {
    title = "Entity Investigation";
    subtitle = view.query.type === "person" ? view.query.identity : `Track #${view.query.trackId}`;
  }

  return (
    <div className="app-shell">
      <Sidebar
        active={activeSection}
        onNavigate={navigate}
        camerasOnline={onlineCount}
        camerasMax={status?.cameras_max ?? MAX_SLOTS}
        activeAlertCount={alertList.filter((a) => a.status === "NEW").length}
        systemLabel={systemLabel}
        systemDotClass={systemDotClass}
      />

      <div className="app-main">
        <TopBar
          title={title}
          subtitle={subtitle}
          status={status}
          connected={statusConnected}
          systemLabel={systemLabel}
          systemDotClass={systemDotClass}
        />

        <div className="app-content">
          {view.kind === "camera" && focusedCamera ? (
            <CameraDetail
              camera={focusedCamera}
              alerts={alertList.filter((a) => a.camera_id === focusedCamera.camera_id)}
              onBack={backToSection}
            />
          ) : view.kind === "investigation" ? (
            <Investigation
              eventId={view.eventId}
              onBack={backToSection}
              onSelectEvent={openInvestigation}
              onSelectTrack={(cameraId, trackId) => openEntity({ type: "track", cameraId, trackId })}
              onSelectIdentity={(identity) => openEntity({ type: "person", identity })}
            />
          ) : view.kind === "entity" ? (
            <EntityInvestigation query={view.query} onBack={backToSection} onSelectEvent={openInvestigation} />
          ) : activeSection === "dashboard" ? (
            <>
              <SystemStatusBar
                status={status}
                connected={statusConnected}
                systemLabel={systemLabel}
                systemDotClass={systemDotClass}
              />
              <GlobalStats detections={detections} />
              <AIEngineStatus status={status} />
              <CameraGrid
                cameras={cameraList}
                alerts={alertList}
                onFocusCamera={focusCamera}
                maxSlots={MAX_SLOTS}
              />
              <div className="main-grid">
                <GlobalAlerts
                  alerts={alertList}
                  onFocusCamera={focusCamera}
                  onInvestigate={openInvestigation}
                  statusFilter={["NEW", "ACKNOWLEDGED"]}
                  title="Active Alerts"
                  limit={5}
                  onViewAll={() => navigate("alerts")}
                />
                <EventTimeline
                  events={recentEvents}
                  onSelectEvent={openInvestigation}
                  title="Recent Events"
                  onViewAll={() => navigate("events")}
                />
              </div>
              <CameraManagement
                cameras={cameras ?? []}
                maxSlots={MAX_SLOTS}
                onFocusCamera={focusCamera}
                onAddClick={() => setShowAddModal(true)}
              />
            </>
          ) : activeSection === "alerts" ? (
            <GlobalAlerts alerts={alertList} onFocusCamera={focusCamera} onInvestigate={openInvestigation} />
          ) : activeSection === "events" ? (
            <EventHistory cameras={cameras ?? []} onSelectEvent={openInvestigation} />
          ) : (
            <ZoneManagement cameras={cameras ?? []} />
          )}
        </div>
      </div>

      {showAddModal && (
        <AddCameraModal
          atLimit={(cameras?.length ?? 0) >= MAX_SLOTS}
          onClose={() => setShowAddModal(false)}
          onAdded={() => setShowAddModal(false)}
        />
      )}

      {!detectionsConnected && !alertsConnected && (
        <div className="conn-banner">Waiting for backend connection — is the FastAPI server running?</div>
      )}
    </div>
  );
}
