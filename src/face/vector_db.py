import os
import sys
import psycopg2
import numpy as np
import cv2
from typing import List, Tuple, Optional, Dict

class PostgresVectorDatabase:
    """A persistent vector database using PostgreSQL for face embeddings and frame crops."""

    def __init__(self, db_uri: str):
        self.db_uri = db_uri
        self._init_db()

    def _get_connection(self):
        """Returns a new database connection."""
        return psycopg2.connect(self.db_uri)

    def _init_db(self):
        """Initializes database tables if they do not exist."""
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cursor:
                    # Table for enrolled reference users
                    cursor.execute("""
                        CREATE TABLE IF NOT EXISTS gallery_embeddings (
                            id SERIAL PRIMARY KEY,
                            identity VARCHAR(255) NOT NULL,
                            embedding BYTEA NOT NULL,
                            frame BYTEA,
                            source_name VARCHAR(255) UNIQUE
                        );
                    """)
                    # Table for flagged unauthorized events (intruders)
                    cursor.execute("""
                        CREATE TABLE IF NOT EXISTS flagged_events (
                            id SERIAL PRIMARY KEY,
                            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            embedding BYTEA NOT NULL,
                            frame BYTEA NOT NULL,
                            track_id INTEGER,
                            video_source VARCHAR(255)
                        );
                    """)
                    conn.commit()
            print("[INFO] PostgreSQL database tables initialized successfully.", file=sys.stderr)
        except Exception as e:
            print(f"[ERROR] Failed to connect or initialize PostgreSQL database: {e}", file=sys.stderr)
            print("[ERROR] Please verify PostgreSQL is running and credentials/database exist.", file=sys.stderr)
            raise e

    def add_user(self, identity: str, embedding: np.ndarray, frame: Optional[np.ndarray], source_name: Optional[str] = None) -> bool:
        """
        Enrolls a user embedding and face crop frame in the database.
        Uses INSERT ON CONFLICT to avoid duplicate reference entries.
        """
        emb_blob = embedding.astype(np.float32).tobytes()
        
        frame_blob = None
        if frame is not None and frame.size > 0:
            success, buffer = cv2.imencode('.jpg', frame)
            if success:
                frame_blob = buffer.tobytes()

        try:
            with self._get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("""
                        INSERT INTO gallery_embeddings (identity, embedding, frame, source_name)
                        VALUES (%s, %s, %s, %s)
                        ON CONFLICT (source_name) 
                        DO UPDATE SET identity = EXCLUDED.identity, embedding = EXCLUDED.embedding, frame = EXCLUDED.frame;
                    """, (identity, psycopg2.Binary(emb_blob), psycopg2.Binary(frame_blob) if frame_blob else None, source_name))
                    conn.commit()
            return True
        except Exception as e:
            print(f"[ERROR] Failed to add user '{identity}' to PostgreSQL database: {e}", file=sys.stderr)
            return False

    def has_user_file(self, source_name: str) -> bool:
        """Checks if a source image has already been enrolled in the database."""
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(
                        "SELECT 1 FROM gallery_embeddings WHERE source_name = %s LIMIT 1;", 
                        (source_name,)
                    )
                    return cursor.fetchone() is not None
        except Exception as e:
            print(f"[ERROR] Failed to query file existence for '{source_name}': {e}", file=sys.stderr)
            return False

    def fetch_all_users(self) -> List[Tuple[str, np.ndarray, Optional[np.ndarray], Optional[str]]]:
        """Fetches all reference users, embeddings, and frame crops from PostgreSQL."""
        users = []
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("SELECT identity, embedding, frame, source_name FROM gallery_embeddings;")
                    rows = cursor.fetchall()
                    for row in rows:
                        identity, emb_bytes, frame_bytes, source_name = row
                        embedding = np.frombuffer(emb_bytes, dtype=np.float32)
                        
                        frame = None
                        if frame_bytes is not None:
                            nparr = np.frombuffer(frame_bytes, dtype=np.uint8)
                            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                        
                        users.append((identity, embedding, frame, source_name))
        except Exception as e:
            print(f"[ERROR] Failed to fetch users from PostgreSQL: {e}", file=sys.stderr)
        return users

    def flag_unauthorized_user(self, embedding: np.ndarray, frame: np.ndarray, track_id: int, video_source: str) -> bool:
        """Stores a flagged intruder event in the database."""
        emb_blob = embedding.astype(np.float32).tobytes()
        
        frame_blob = b""
        if frame is not None and frame.size > 0:
            success, buffer = cv2.imencode('.jpg', frame)
            if success:
                frame_blob = buffer.tobytes()

        try:
            with self._get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("""
                        INSERT INTO flagged_events (embedding, frame, track_id, video_source)
                        VALUES (%s, %s, %s, %s);
                    """, (psycopg2.Binary(emb_blob), psycopg2.Binary(frame_blob), track_id, video_source))
                    conn.commit()
            return True
        except Exception as e:
            print(f"[ERROR] Failed to store flagged event for track #{track_id}: {e}", file=sys.stderr)
            return False

    def fetch_flagged_events(self) -> List[Tuple[int, str, np.ndarray, Optional[np.ndarray], int, str]]:
        """Fetches all flagged intruder events from PostgreSQL."""
        events = []
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("SELECT id, timestamp, embedding, frame, track_id, video_source FROM flagged_events ORDER BY id DESC;")
                    rows = cursor.fetchall()
                    for row in rows:
                        ev_id, timestamp, emb_bytes, frame_bytes, track_id, video_source = row
                        embedding = np.frombuffer(emb_bytes, dtype=np.float32)
                        
                        frame = None
                        if frame_bytes:
                            nparr = np.frombuffer(frame_bytes, dtype=np.uint8)
                            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                        
                        events.append((ev_id, str(timestamp), embedding, frame, track_id, video_source))
        except Exception as e:
            print(f"[ERROR] Failed to fetch flagged events from PostgreSQL: {e}", file=sys.stderr)
        return events
