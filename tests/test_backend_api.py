import os
import sys
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

    def test_global_events_contract(self):
        resp = self.client.get("/events")
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertIsInstance(body, list)
        for alert in body:
            for key in ("alert_id", "camera_id", "camera_name", "severity", "title", "status", "timestamp", "track_id"):
                self.assertIn(key, alert)

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


if __name__ == "__main__":
    unittest.main()
