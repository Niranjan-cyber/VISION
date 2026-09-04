import { useState } from "react";
import Sidebar, { type NavSection } from "./components/Sidebar";
import TopBar from "./components/TopBar";
import SystemStatusBar from "./components/SystemStatusBar";
import AIEngineStatus from "./components/AIEngineStatus";
import CameraGrid from "./components/CameraGrid";
import CameraDetail from "./components/CameraDetail";
import GlobalAlerts from "./components/GlobalAlerts";
import GlobalStats from "./components/GlobalStats";
import EventTimeline from "./components/EventTimeline";
import CameraManagement from "./components/CameraManagement";
import AddCameraModal from "./components/AddCameraModal";
import {
  findCameraDetections,
  useCameraList,
  useGlobalDetections,
  useGlobalEvents,
  useGlobalStatus,
} from "./api";
import type { GlobalStatus } from "./types";

const MAX_SLOTS = 4;

const SECTION_META: Record<NavSection, { title: string; subtitle: string }> = {
  dashboard: { title: "Dashboard", subtitle: "Live surveillance feeds from connected border locations" },
  alerts: { title: "Alerts", subtitle: "AI-generated security events across the surveillance network" },
  events: { title: "Events", subtitle: "Chronological event log across every camera" },
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

export default function App() {
  const { data: detections, connected: detectionsConnected } = useGlobalDetections();
  const { data: alerts } = useGlobalEvents();
  const { data: status, connected: statusConnected } = useGlobalStatus();
  const { data: cameras } = useCameraList();

  const [activeSection, setActiveSection] = useState<NavSection>("dashboard");
  const [focusedCameraId, setFocusedCameraId] = useState<string | null>(null);
  const [showAddModal, setShowAddModal] = useState(false);

  const cameraList = detections?.cameras ?? [];
  const alertList = alerts ?? [];
  const focusedCamera = focusedCameraId ? findCameraDetections(detections, focusedCameraId) : null;
  const { label: systemLabel, dotClass: systemDotClass } = systemStatus(status, statusConnected);
  const onlineCount = status ? Object.values(status.cameras).filter((c) => c.status === "online").length : 0;

  function focusCamera(cameraId: string) {
    setFocusedCameraId(cameraId);
  }

  return (
    <div className="app-shell">
      <Sidebar
        active={activeSection}
        onNavigate={(section) => {
          setActiveSection(section);
          setFocusedCameraId(null);
        }}
        camerasOnline={onlineCount}
        camerasMax={status?.cameras_max ?? MAX_SLOTS}
        activeAlertCount={alertList.length}
        systemLabel={systemLabel}
        systemDotClass={systemDotClass}
      />

      <div className="app-main">
        <TopBar
          title={focusedCamera ? focusedCamera.camera_name : SECTION_META[activeSection].title}
          subtitle={focusedCamera ? focusedCamera.camera_id : SECTION_META[activeSection].subtitle}
          status={status}
          connected={statusConnected}
          systemLabel={systemLabel}
          systemDotClass={systemDotClass}
        />

        <div className="app-content">
          {focusedCamera ? (
            <CameraDetail
              camera={focusedCamera}
              alerts={alertList.filter((a) => a.camera_id === focusedCamera.camera_id)}
              onBack={() => setFocusedCameraId(null)}
            />
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
              <CameraManagement
                cameras={cameras ?? []}
                maxSlots={MAX_SLOTS}
                onFocusCamera={focusCamera}
                onAddClick={() => setShowAddModal(true)}
              />
            </>
          ) : activeSection === "alerts" ? (
            <GlobalAlerts alerts={alertList} onFocusCamera={focusCamera} />
          ) : (
            <EventTimeline alerts={alertList} />
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

      {!detectionsConnected && (
        <div className="conn-banner">Waiting for backend connection — is the FastAPI server running?</div>
      )}
    </div>
  );
}
