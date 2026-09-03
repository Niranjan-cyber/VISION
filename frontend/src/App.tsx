import { useState } from "react";
import Header from "./components/Header";
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

const MAX_SLOTS = 4;

export default function App() {
  const { data: detections, connected: detectionsConnected } = useGlobalDetections();
  const { data: alerts } = useGlobalEvents();
  const { data: status, connected: statusConnected } = useGlobalStatus();
  const { data: cameras } = useCameraList();

  const [focusedCameraId, setFocusedCameraId] = useState<string | null>(null);
  const [showAddModal, setShowAddModal] = useState(false);

  const cameraList = detections?.cameras ?? [];
  const alertList = alerts ?? [];
  const focusedCamera = focusedCameraId ? findCameraDetections(detections, focusedCameraId) : null;

  return (
    <div className="app-shell">
      <Header status={status} connected={statusConnected} />

      {focusedCamera ? (
        <CameraDetail
          camera={focusedCamera}
          alerts={alertList.filter((a) => a.camera_id === focusedCamera.camera_id)}
          onBack={() => setFocusedCameraId(null)}
        />
      ) : (
        <>
          <CameraGrid
            cameras={cameraList}
            alerts={alertList}
            onFocusCamera={setFocusedCameraId}
            maxSlots={MAX_SLOTS}
          />

          <GlobalAlerts alerts={alertList} onFocusCamera={setFocusedCameraId} />

          <GlobalStats detections={detections} />

          <EventTimeline alerts={alertList} />

          <CameraManagement
            cameras={cameras ?? []}
            maxSlots={MAX_SLOTS}
            onFocusCamera={setFocusedCameraId}
            onAddClick={() => setShowAddModal(true)}
          />
        </>
      )}

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
