import os
import sys
import tempfile
import time
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Boot with just the golden camera — fast, and exercises the exact same
# contract multi-camera boots use, just with one camera instead of four.
os.environ["VISION_CAMERA_COUNT"] = "1"
os.environ["VISION_DEMO_VIDEO"] = "data/videos/shreyas1.mp4"
os.environ["VISION_DEMO_ZONES"] = "configs/zones_demo.yaml"
os.environ["VISION_LOITERING_DURATION"] = "3.0"
os.environ["VISION_ENABLE_ANPR"] = "false"
# Isolated Phase 3 event/alert/zone database — never writes test data into
# the shared demo database (data/vision.db).
os.environ["VISION_DB_PATH"] = os.path.join(tempfile.mkdtemp(prefix="vision_test_backend_api_db_"), "test.db")
os.environ["VISION_SNAPSHOT_DIR"] = tempfile.mkdtemp(prefix="vision_test_backend_api_snapshots_")

from fastapi.testclient import TestClient

from backend.main import app

STARTUP_TIMEOUT_S = 45


class TestBackendAPI(unittest.TestCase):
    """
    Minimal MVP-level validation that the FastAPI layer actually serves the
    live multi-camera pipeline. Boots the real background CAM-01 session
    once for the whole test class; camera-management HTTP flows (add/
    restart/remove) run against additional disposable upload cameras so
    they never disturb the golden CAM-01 session other tests depend on.
    """

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        cls.client.__enter__()  # runs the lifespan startup, starts CAM-01's worker thread

        deadline = time.time() + STARTUP_TIMEOUT_S
        status = {}
        while time.time() < deadline:
            resp = cls.client.get("/status")
            status = resp.json()
            cam01 = status.get("cameras", {}).get("CAM-01", {})
            if cam01.get("status") == "online":
                break
            time.sleep(1)
        else:
            raise RuntimeError(f"CAM-01 did not become ready within {STARTUP_TIMEOUT_S}s: {status}")

        # A couple more seconds so /detections has non-empty person data.
        time.sleep(3)

    @classmethod
    def tearDownClass(cls):
        cls.client.__exit__(None, None, None)

    # ------------------------------------------------------------------
    # Global / health
    # ------------------------------------------------------------------
    def test_health(self):
        resp = self.client.get("/health")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json(), {"status": "ok"})

    def test_global_status_reports_real_camera_state(self):
        resp = self.client.get("/status")
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertIn("cameras_active", body)
        self.assertIn("cameras_max", body)
        self.assertEqual(body["cameras_max"], 4)
        self.assertIn("CAM-01", body["cameras"])
        self.assertEqual(body["cameras"]["CAM-01"]["status"], "online")

    def test_global_status_reports_real_ai_engine_devices(self):
        resp = self.client.get("/status")
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertIn("ai_engine", body)
        engine = body["ai_engine"]
        for key in ("yolo_device", "face_recognition_device", "yunet_device", "tracking_device", "event_engine_device", "ai_fps"):
            self.assertIn(key, engine)
        self.assertIn(engine["yolo_device"], ("CUDA", "CPU"))
        self.assertIn(engine["face_recognition_device"], ("CUDA", "CPU"))
        self.assertEqual(engine["yunet_device"], "CPU")  # never claims GPU YuNet doesn't have
        self.assertEqual(engine["tracking_device"], "CPU")
        self.assertEqual(engine["event_engine_device"], "CPU")
        self.assertGreater(engine["ai_fps"], 0)

    def test_global_detections_contract_and_nonempty_output(self):
        resp = self.client.get("/detections")
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertIn("cameras", body)
        self.assertIn("statistics", body)
        cam01 = next(c for c in body["cameras"] if c["camera_id"] == "CAM-01")
        self.assertGreater(len(cam01["persons"]), 0, "golden demo video should have a tracked person by now")
        person = cam01["persons"][0]
        for key in ("track_id", "identity", "face_similarity", "bbox", "confidence", "zone"):
            self.assertIn(key, person)

    def test_global_alerts_contract(self):
        # Phase 3: GET /events used to serve this live-alert-feed role
        # (Phase 2). It now means historical event search (see
        # test_events_history_contract below) — the operator-facing,
        # lifecycle-managed alert feed moved to its own endpoint, GET
        # /alerts, backed by the persistent alert store instead of each
        # camera's in-memory EventEngine.active_alerts.
        resp = self.client.get("/alerts")
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertIsInstance(body, list)
        for alert in body:
            for key in ("alert_id", "camera_id", "camera_name", "severity", "title", "status", "timestamp", "track_id"):
                self.assertIn(key, alert)

    def test_events_history_contract(self):
        resp = self.client.get("/events")
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertIsInstance(body, list)
        for event in body:
            for key in ("event_id", "camera_id", "camera_name", "event_type", "severity", "timestamp", "created_at", "description", "has_snapshot"):
                self.assertIn(key, event)

    # ------------------------------------------------------------------
    # Camera list / per-camera endpoints (golden CAM-01, read-only)
    # ------------------------------------------------------------------
    def test_cameras_list_includes_golden_camera(self):
        resp = self.client.get("/cameras")
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(len(body), 1)
        self.assertEqual(body[0]["camera_id"], "CAM-01")
        self.assertEqual(body[0]["status"], "online")

    def test_camera_status_endpoint(self):
        resp = self.client.get("/cameras/CAM-01/status")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["status"], "online")

    def test_camera_detections_endpoint(self):
        resp = self.client.get("/cameras/CAM-01/detections")
        self.assertEqual(resp.status_code, 200)
        self.assertIn("persons", resp.json())

    def test_camera_events_endpoint(self):
        resp = self.client.get("/cameras/CAM-01/events")
        self.assertEqual(resp.status_code, 200)
        self.assertIsInstance(resp.json(), list)

    def test_unknown_camera_returns_404(self):
        for path in ("/cameras/CAM-99/status", "/cameras/CAM-99/detections", "/cameras/CAM-99/events"):
            resp = self.client.get(path)
            self.assertEqual(resp.status_code, 404, path)

    def test_restart_unknown_camera_returns_404(self):
        resp = self.client.post("/cameras/CAM-99/restart")
        self.assertEqual(resp.status_code, 404)

    def test_remove_unknown_camera_returns_404(self):
        resp = self.client.delete("/cameras/CAM-99")
        self.assertEqual(resp.status_code, 404)

    # ------------------------------------------------------------------
    # Camera management HTTP flow (disposable upload camera — never CAM-01)
    # ------------------------------------------------------------------
    def test_add_camera_via_upload_then_restart_then_remove(self):
        with open("data/videos/test.mp4", "rb") as f:
            resp = self.client.post(
                "/cameras",
                data={"camera_name": "HTTP Test Cam"},
                files={"video": ("test.mp4", f, "video/mp4")},
            )
        self.assertEqual(resp.status_code, 200, resp.text)
        cam = resp.json()
        camera_id = cam["camera_id"]
        self.assertEqual(cam["camera_name"], "HTTP Test Cam")
        self.assertIn(cam["status"], ("starting", "online"))

        try:
            # Restart should be accepted even while still starting.
            resp = self.client.post(f"/cameras/{camera_id}/restart")
            self.assertEqual(resp.status_code, 200)

            resp = self.client.get("/cameras")
            ids = [c["camera_id"] for c in resp.json()]
            self.assertIn(camera_id, ids)
        finally:
            resp = self.client.delete(f"/cameras/{camera_id}")
            self.assertEqual(resp.status_code, 200)

        resp = self.client.get("/cameras")
        ids = [c["camera_id"] for c in resp.json()]
        self.assertNotIn(camera_id, ids)

    def test_add_camera_rejects_unsupported_extension(self):
        with open("README.md", "rb") as f:
            resp = self.client.post(
                "/cameras",
                data={"camera_name": "Bad Ext"},
                files={"video": ("README.md", f, "text/plain")},
            )
        self.assertEqual(resp.status_code, 400)

    # ------------------------------------------------------------------
    # Phase 3 — Alert management / Historical events / Investigation / Zones
    # (all against CAM-01's real, already-running golden session)
    # ------------------------------------------------------------------
    def _wait_for_cam01_event(self, timeout=30):
        deadline = time.time() + timeout
        while time.time() < deadline:
            resp = self.client.get("/events", params={"camera_id": "CAM-01", "limit": 5})
            events = resp.json()
            if events:
                return events
            time.sleep(1)
        self.fail("CAM-01 did not persist any event within the timeout")

    def test_events_history_filters_and_pagination(self):
        events = self._wait_for_cam01_event()
        event_id = events[0]["event_id"]

        resp = self.client.get("/events", params={"camera_id": "CAM-01"})
        self.assertEqual(resp.status_code, 200)
        self.assertTrue(all(e["camera_id"] == "CAM-01" for e in resp.json()))

        resp = self.client.get("/events", params={"camera_id": "CAM-99"})
        self.assertEqual(resp.json(), [])

        resp = self.client.get("/events", params={"event_type": "INTRUSION"})
        self.assertTrue(all(e["event_type"] == "INTRUSION" for e in resp.json()))

        resp = self.client.get("/events", params={"limit": 1})
        self.assertLessEqual(len(resp.json()), 1)

        resp = self.client.get(f"/events/{event_id}")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["event_id"], event_id)

        resp = self.client.get("/events/does-not-exist")
        self.assertEqual(resp.status_code, 404)

    def test_event_snapshot_served_when_present(self):
        events = self._wait_for_cam01_event()
        with_snapshot = next((e for e in events if e["has_snapshot"]), None)
        self.assertIsNotNone(with_snapshot, "expected at least one persisted event to have a snapshot")
        resp = self.client.get(f"/events/{with_snapshot['event_id']}/snapshot")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.headers["content-type"], "image/jpeg")

    def test_event_snapshot_missing_returns_404_not_fake_image(self):
        resp = self.client.get("/events/does-not-exist/snapshot")
        self.assertEqual(resp.status_code, 404)

    def test_alert_lifecycle_new_to_acknowledged_to_resolved(self):
        self._wait_for_cam01_event()
        resp = self.client.get("/alerts", params={"camera_id": "CAM-01", "status": "NEW"})
        alerts = resp.json()
        self.assertTrue(alerts, "expected at least one NEW alert for CAM-01 by now")
        alert_id = alerts[0]["alert_id"]

        resp = self.client.post(f"/alerts/{alert_id}/acknowledge")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["status"], "ACKNOWLEDGED")
        self.assertIsNotNone(resp.json()["acknowledged_at"])

        # Acknowledging twice is a nonsensical transition (ACKNOWLEDGED -> ACKNOWLEDGED).
        resp = self.client.post(f"/alerts/{alert_id}/acknowledge")
        self.assertEqual(resp.status_code, 409)

        resp = self.client.post(f"/alerts/{alert_id}/resolve")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["status"], "RESOLVED")

        # RESOLVED is terminal.
        resp = self.client.post(f"/alerts/{alert_id}/acknowledge")
        self.assertEqual(resp.status_code, 409)

        # The camera's own live alert list must reflect it too (not just the DB).
        resp = self.client.get("/cameras/CAM-01/events")
        matching = [a for a in resp.json() if a["alert_id"] == alert_id]
        if matching:  # camera-scoped feed only keeps the most recent N alerts
            self.assertEqual(matching[0]["status"], "RESOLVED")

    def test_acknowledge_unknown_alert_returns_404(self):
        resp = self.client.post("/alerts/does-not-exist/acknowledge")
        self.assertEqual(resp.status_code, 404)

    def test_investigate_event_includes_related_events(self):
        events = self._wait_for_cam01_event()
        event_id = events[0]["event_id"]
        resp = self.client.get(f"/investigations/event/{event_id}")
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body["event"]["event_id"], event_id)
        self.assertIn("alert", body)
        self.assertIn("related_events", body)
        self.assertIsInstance(body["related_events"], list)

    def test_investigate_recognized_person(self):
        self._wait_for_cam01_event()
        resp = self.client.get("/investigations/person/Shreyas_Chavan")
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body["identity"], "Shreyas_Chavan")
        self.assertTrue(body["recognized"])
        self.assertIn("CAM-01", body["cameras"])

    def test_investigate_person_rejects_unknown_as_not_a_real_identity(self):
        resp = self.client.get("/investigations/person/UNKNOWN")
        self.assertEqual(resp.status_code, 400)

    def test_investigate_person_with_no_events_returns_404(self):
        resp = self.client.get("/investigations/person/Nobody_Recognized_Ever")
        self.assertEqual(resp.status_code, 404)

    def test_investigate_track(self):
        events = self._wait_for_cam01_event()
        track_id = events[0]["track_id"]
        resp = self.client.get(f"/investigations/track/CAM-01/{track_id}")
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body["camera_id"], "CAM-01")
        self.assertEqual(body["track_id"], track_id)

    def test_investigate_track_with_no_events_returns_404(self):
        resp = self.client.get("/investigations/track/CAM-01/999999")
        self.assertEqual(resp.status_code, 404)

    def test_zones_seeded_from_golden_demo_yaml(self):
        resp = self.client.get("/zones", params={"camera_id": "CAM-01"})
        self.assertEqual(resp.status_code, 200)
        zones = resp.json()
        self.assertTrue(zones, "CAM-01's YAML zone should have been seeded into the zone store")
        self.assertTrue(all(z["camera_id"] == "CAM-01" for z in zones))

    def test_zone_crud_lifecycle(self):
        resp = self.client.post(
            "/zones",
            json={"camera_id": "CAM-01", "name": "Test Zone", "type": "restricted",
                  "polygon": [[10, 10], [100, 10], [100, 100], [10, 100]]},
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        zone = resp.json()
        zone_id = zone["id"]
        self.assertEqual(zone["name"], "Test Zone")
        self.assertTrue(zone["enabled"])

        resp = self.client.get(f"/zones/{zone_id}")
        self.assertEqual(resp.status_code, 200)

        resp = self.client.put(f"/zones/{zone_id}", json={"name": "Renamed Zone", "enabled": False})
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["name"], "Renamed Zone")
        self.assertFalse(resp.json()["enabled"])

        resp = self.client.delete(f"/zones/{zone_id}")
        self.assertEqual(resp.status_code, 200)

        resp = self.client.get(f"/zones/{zone_id}")
        self.assertEqual(resp.status_code, 404)

    def test_zone_creation_rejects_invalid_geometry(self):
        resp = self.client.post(
            "/zones",
            json={"camera_id": "CAM-01", "name": "Bad", "type": "restricted", "polygon": [[0, 0], [1, 1]]},
        )
        self.assertEqual(resp.status_code, 400)

    def test_zone_creation_rejects_unknown_camera(self):
        resp = self.client.post(
            "/zones",
            json={"camera_id": "CAM-99", "name": "Orphan", "type": "restricted",
                  "polygon": [[0, 0], [10, 0], [10, 10]]},
        )
        self.assertEqual(resp.status_code, 404)

    def test_no_zone_configured_for_camera_without_zones(self):
        # A freshly-uploaded camera has no zones_path — must show zero zones,
        # never a fabricated one.
        with open("data/videos/test.mp4", "rb") as f:
            resp = self.client.post(
                "/cameras",
                data={"camera_name": "No Zone Cam"},
                files={"video": ("test.mp4", f, "video/mp4")},
            )
        camera_id = resp.json()["camera_id"]
        try:
            resp = self.client.get("/zones", params={"camera_id": camera_id})
            self.assertEqual(resp.status_code, 200)
            self.assertEqual(resp.json(), [])
        finally:
            self.client.delete(f"/cameras/{camera_id}")


if __name__ == "__main__":
    unittest.main()
