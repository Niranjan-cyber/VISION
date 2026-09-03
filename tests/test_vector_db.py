import unittest
from unittest.mock import MagicMock, patch
import numpy as np
import cv2

from src.core.types import FaceEmbedding
from src.face.vector_db import PostgresVectorDatabase
from src.face.gallery import FaceGallery


class TestPostgresVectorDatabase(unittest.TestCase):
    """Unit test suite for PostgresVectorDatabase functionality with psycopg2 mocking."""

    @patch("psycopg2.connect")
    def setUp(self, mock_connect):
        # Mock connection and cursor
        self.mock_conn = MagicMock()
        self.mock_cursor = MagicMock()
        self.mock_conn.cursor.return_value.__enter__.return_value = self.mock_cursor
        mock_connect.return_value = self.mock_conn

        self.db_uri = "postgresql://mock_user:mock_pass@localhost:5432/mock_db"
        self.db = PostgresVectorDatabase(self.db_uri)

    def test_1_db_initialization_creates_tables(self):
        """TEST 1: DB initialization executes table creation commands."""
        # The database initialization runs inside setUp (PostgresVectorDatabase(db_uri))
        # Verify the table creation commands were executed
        self.assertEqual(self.mock_cursor.execute.call_count, 2)
        calls = [c[0][0] for c in self.mock_cursor.execute.call_args_list]
        self.assertTrue(any("CREATE TABLE IF NOT EXISTS gallery_embeddings" in call for call in calls))
        self.assertTrue(any("CREATE TABLE IF NOT EXISTS flagged_events" in call for call in calls))

    @patch("psycopg2.connect")
    def test_2_add_user_serializes_inputs(self, mock_connect):
        """TEST 2: add_user converts embeddings and frames to correct binary formats."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_connect.return_value = mock_conn

        db = PostgresVectorDatabase(self.db_uri)
        mock_cursor.execute.reset_mock()

        # Dummy inputs
        identity = "user_007"
        embedding = np.ones(512, dtype=np.float32)
        frame = np.zeros((50, 50, 3), dtype=np.uint8)
        source_name = "test_path/image.jpg"

        success = db.add_user(identity, embedding, frame, source_name)
        self.assertTrue(success)

        # Verify executing query
        mock_cursor.execute.assert_called_once()
        query, params = mock_cursor.execute.call_args[0]
        self.assertIn("INSERT INTO gallery_embeddings", query)
        self.assertEqual(params[0], identity)
        # Check binary serialization of embedding (psycopg2.Binary wraps bytes)
        self.assertIsInstance(params[1].adapted, bytes)
        self.assertEqual(len(params[1].adapted), 512 * 4)  # 2048 bytes
        # Check binary serialization of frame
        self.assertIsInstance(params[2].adapted, bytes)
        self.assertEqual(params[3], source_name)

    @patch("psycopg2.connect")
    def test_3_has_user_file_queries_db(self, mock_connect):
        """TEST 3: has_user_file queries the database and returns boolean correctly."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_connect.return_value = mock_conn

        db = PostgresVectorDatabase(self.db_uri)

        # Case 1: Image exists
        mock_cursor.fetchone.return_value = (1,)
        self.assertTrue(db.has_user_file("existing.jpg"))

        # Case 2: Image does not exist
        mock_cursor.fetchone.return_value = None
        self.assertFalse(db.has_user_file("missing.jpg"))

    @patch("psycopg2.connect")
    def test_4_fetch_all_users_deserializes_correctly(self, mock_connect):
        """TEST 4: fetch_all_users reads and decodes embeddings and JPEG frames."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_connect.return_value = mock_conn

        db = PostgresVectorDatabase(self.db_uri)

        # Generate fake DB row values
        identity = "user_abc"
        embedding_orig = np.arange(512, dtype=np.float32)
        emb_bytes = embedding_orig.tobytes()
        
        frame_orig = np.zeros((10, 10, 3), dtype=np.uint8)
        _, buffer = cv2.imencode('.jpg', frame_orig)
        frame_bytes = buffer.tobytes()
        source_name = "path/abc.jpg"

        mock_cursor.fetchall.return_value = [(identity, emb_bytes, frame_bytes, source_name)]

        users = db.fetch_all_users()
        self.assertEqual(len(users), 1)
        
        ret_id, ret_emb, ret_frame, ret_src = users[0]
        self.assertEqual(ret_id, identity)
        self.assertTrue(np.array_equal(ret_emb, embedding_orig))
        self.assertIsNotNone(ret_frame)
        self.assertEqual(ret_frame.shape, (10, 10, 3))
        self.assertEqual(ret_src, source_name)

    @patch("psycopg2.connect")
    def test_5_flag_unauthorized_user_saves_event(self, mock_connect):
        """TEST 5: flag_unauthorized_user stores unrecognized person embedding and frame."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_connect.return_value = mock_conn

        db = PostgresVectorDatabase(self.db_uri)
        mock_cursor.execute.reset_mock()

        embedding = np.zeros(512, dtype=np.float32)
        frame = np.ones((40, 40, 3), dtype=np.uint8)
        track_id = 99
        video_source = "data/videos/test.mp4"

        success = db.flag_unauthorized_user(embedding, frame, track_id, video_source)
        self.assertTrue(success)

        mock_cursor.execute.assert_called_once()
        query, params = mock_cursor.execute.call_args[0]
        self.assertIn("INSERT INTO flagged_events", query)
        self.assertIsInstance(params[0].adapted, bytes)
        self.assertIsInstance(params[1].adapted, bytes)
        self.assertEqual(params[2], track_id)
        self.assertEqual(params[3], video_source)

    @patch("psycopg2.connect")
    def test_6_integration_with_gallery_initialization(self, mock_connect):
        """TEST 6: FaceGallery correctly instantiates PostgresVectorDatabase with URI."""
        gallery = FaceGallery(db_uri=self.db_uri)
        self.assertIsNotNone(gallery.db)
        self.assertEqual(gallery.db.db_uri, self.db_uri)


if __name__ == "__main__":
    unittest.main()
