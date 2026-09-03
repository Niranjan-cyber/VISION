import os
import sys
import time
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Point the backend at the same golden demo config used everywhere else,
# regardless of what env vars happen to be set in the test environment.
os.environ["VISION_DEMO_VIDEO"] = "data/videos/shreyas1.mp4"
os.environ["VISION_DEMO_ZONES"] = "configs/zones_demo.yaml"
os.environ["VISION_LOITERING_DURATION"] = "3.0"
os.environ["VISION_ENABLE_ANPR"] = "false"

from fastapi.testclient import TestClient

from backend.main import app

STARTUP_TIMEOUT_S = 45


class TestBackendAPI(unittest.TestCase):
    """
    Minimal MVP-level validation that the FastAPI layer actually serves a
    live pipeline, not that every edge case is covered. Boots the real
    background PipelineSession once for the whole test class.
    """

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        cls.client.__enter__()  # runs the lifespan startup, starts the worker thread

        deadline = time.time() + STARTUP_TIMEOUT_S
        status = {}
        while time.time() < deadline:
            resp = cls.client.get("/status")
            status = resp.json()
            if status.get("video") and status.get("detection") and status.get("face_id"):
                break
            time.sleep(1)
        else:
            raise RuntimeError(f"Backend pipeline did not become ready within {STARTUP_TIMEOUT_S}s: {status}")

        # Give it a couple more seconds to process a handful of real frames
        # so /detections has non-empty person data to assert against.
        time.sleep(3)

    @classmethod
    def tearDownClass(cls):
        cls.client.__exit__(None, None, None)

    def test_health(self):
        resp = self.client.get("/health")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json(), {"status": "ok"})

    def test_status_reports_real_subsystems(self):
        resp = self.client.get("/status")
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        for key in ("video", "detection", "tracking", "face_id", "anpr", "events"):
            self.assertIn(key, body)
        self.assertFalse(body["anpr"])  # VISION_ENABLE_ANPR=false

    def test_detections_contract_and_nonempty_output(self):
        resp = self.client.get("/detections")
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        for key in ("timestamp", "frame_id", "persons", "vehicles", "statistics", "anpr_enabled", "status"):
            self.assertIn(key, body)
        self.assertFalse(body["anpr_enabled"])
        self.assertGreater(len(body["persons"]), 0, "golden demo video should have a tracked person by now")
        person = body["persons"][0]
        for key in ("track_id", "identity", "face_similarity", "bbox", "confidence", "zone"):
            self.assertIn(key, person)
        self.assertEqual(len(person["bbox"]), 4)

    def test_events_contract(self):
        resp = self.client.get("/events")
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertIsInstance(body, list)
        for alert in body:
            for key in ("alert_id", "severity", "title", "message", "status", "timestamp", "track_id"):
                self.assertIn(key, alert)


if __name__ == "__main__":
    unittest.main()
