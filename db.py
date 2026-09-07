"""Database management module using SQLite WAL mode."""

import sqlite3
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime, timezone
from config import settings


class Database:
    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or settings.DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.init_db()

    def get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        conn.row_factory = sqlite3.Row
        # Enable WAL mode and synchronous normal for high performance and durability
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
        return conn

    def init_db(self) -> None:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            # Videos table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS videos (
                    video_id TEXT PRIMARY KEY,
                    url TEXT NOT NULL,
                    title TEXT,
                    channel_title TEXT,
                    duration_sec INTEGER DEFAULT 0,
                    published_at TEXT,
                    processed_at TEXT NOT NULL,
                    status TEXT NOT NULL,
                    error_message TEXT
                );
                """
            )

            # Clips table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS clips (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    video_id TEXT NOT NULL,
                    start_sec REAL NOT NULL,
                    end_sec REAL NOT NULL,
                    duration_sec REAL NOT NULL,
                    hook_score INTEGER DEFAULT 0,
                    title TEXT,
                    description TEXT,
                    hashtags TEXT,
                    rendered_path TEXT,
                    youtube_status TEXT DEFAULT 'pending',
                    youtube_video_id TEXT,
                    tiktok_status TEXT DEFAULT 'pending',
                    tiktok_publish_id TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (video_id) REFERENCES videos (video_id) ON DELETE CASCADE
                );
                """
            )

            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_videos_status ON videos (status);"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_clips_video_id ON clips (video_id);"
            )
            conn.commit()

    def is_video_processed(self, video_id: str) -> bool:
        """Check if video was already processed or is currently processing."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT 1 FROM videos WHERE video_id = ? AND status IN ('completed', 'processing', 'skipped')",
                (video_id,),
            )
            return cursor.fetchone() is not None

    def record_video(
        self,
        video_id: str,
        url: str,
        title: str = "",
        channel_title: str = "",
        duration_sec: int = 0,
        published_at: str = "",
        status: str = "processing",
    ) -> None:
        now_utc = datetime.now(timezone.utc).isoformat()
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO videos (video_id, url, title, channel_title, duration_sec, published_at, processed_at, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(video_id) DO UPDATE SET
                    status=excluded.status,
                    processed_at=excluded.processed_at;
                """,
                (
                    video_id,
                    url,
                    title,
                    channel_title,
                    duration_sec,
                    published_at,
                    now_utc,
                    status,
                ),
            )
            conn.commit()

    def update_video_status(
        self, video_id: str, status: str, error_message: Optional[str] = None
    ) -> None:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE videos SET status = ?, error_message = ? WHERE video_id = ?",
                (status, error_message, video_id),
            )
            conn.commit()

    def record_clip(
        self,
        video_id: str,
        start_sec: float,
        end_sec: float,
        hook_score: int,
        title: str,
        description: str,
        hashtags: str,
        rendered_path: str,
    ) -> int:
        now_utc = datetime.now(timezone.utc).isoformat()
        duration_sec = end_sec - start_sec
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO clips (
                    video_id, start_sec, end_sec, duration_sec, hook_score,
                    title, description, hashtags, rendered_path, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    video_id,
                    start_sec,
                    end_sec,
                    duration_sec,
                    hook_score,
                    title,
                    description,
                    hashtags,
                    rendered_path,
                    now_utc,
                ),
            )
            clip_id = cursor.lastrowid
            conn.commit()
            return clip_id

    def update_clip_upload(
        self,
        clip_id: int,
        platform: str,
        status: str,
        remote_id: Optional[str] = None,
    ) -> None:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            if platform.lower() == "youtube":
                cursor.execute(
                    "UPDATE clips SET youtube_status = ?, youtube_video_id = ? WHERE id = ?",
                    (status, remote_id, clip_id),
                )
            elif platform.lower() == "tiktok":
                cursor.execute(
                    "UPDATE clips SET tiktok_status = ?, tiktok_publish_id = ? WHERE id = ?",
                    (status, remote_id, clip_id),
                )
            conn.commit()

    def get_stats(self) -> Dict[str, Any]:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM videos")
            total_videos = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM videos WHERE status = 'completed'")
            completed_videos = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM videos WHERE status = 'failed'")
            failed_videos = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM videos WHERE status = 'processing'")
            processing_videos = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM clips")
            total_clips = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM clips WHERE youtube_status = 'success'")
            youtube_uploaded = cursor.fetchone()[0]

            cursor.execute("SELECT AVG(hook_score) FROM clips")
            avg_hook_row = cursor.fetchone()[0]
            avg_hook_score = round(float(avg_hook_row), 1) if avg_hook_row is not None else 0.0

            return {
                "total_videos": total_videos,
                "completed_videos": completed_videos,
                "failed_videos": failed_videos,
                "processing_videos": processing_videos,
                "total_clips": total_clips,
                "youtube_uploaded": youtube_uploaded,
                "avg_hook_score": avg_hook_score,
            }

    def update_clip_rendered_path(self, clip_id: int, new_path: str) -> None:
        """Update rendered_path after upload cleanup (e.g. [UPLOADED_AND_CLEANED])."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE clips SET rendered_path = ? WHERE id = ?",
                (new_path, clip_id),
            )
            conn.commit()

    def get_recent_clips(self, limit: int = 50) -> List[Dict[str, Any]]:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT c.*, v.title as video_title, v.channel_title
                FROM clips c
                LEFT JOIN videos v ON c.video_id = v.video_id
                ORDER BY c.id DESC
                LIMIT ?
                """,
                (limit,),
            )
            return [dict(row) for row in cursor.fetchall()]

    def get_recent_videos(self, limit: int = 50) -> List[Dict[str, Any]]:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT * FROM videos
                ORDER BY processed_at DESC
                LIMIT ?
                """,
                (limit,),
            )
            return [dict(row) for row in cursor.fetchall()]

    def checkpoint_wal(self) -> None:
        """Truncates WAL journal to keep SQLite files minimal and clean."""
        with self.get_connection() as conn:
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE);")


db = Database()
