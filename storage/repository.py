"""SQLite Storage Repository for Auto Short Generator Phase D (Blueprint Bab 18).

Implements persistent database management for:
- videos (tracks 11 happy states and 6 terminal reject states)
- candidates (semantic scored clip candidates)
- renders (rendered 9:16 short video artifacts)
- uploads (YouTube and platform upload status)

Database location: data/app_v3.db (with WAL mode, foreign keys, and indexes).
State transitions are strictly validated against pipeline.state_machine.
"""

import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field

from pipeline.state_machine import (
    PipelineStatus,
    parse_state,
    validate_transition,
    record_transition,
    is_terminal,
    TERMINAL_STATES,
    TERMINAL_REJECT_STATES,
    InvalidStateTransitionError,
    GuardInvariantError,
)

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = Path("/root/projects/auto-short-generator-v3/data/app_v3.db")


def utc_now_iso() -> str:
    """Returns current UTC timestamp in ISO-8601 format."""
    return datetime.now(timezone.utc).isoformat()


# ==============================================================================
# Pydantic Data Models
# ==============================================================================

class VideoRecord(BaseModel):
    """Database entity representing a source video."""
    video_id: str
    url: str
    title: str = ""
    channel_title: str = ""
    duration_sec: float = 0.0
    published_at: str = ""
    status: PipelineStatus = PipelineStatus.DISCOVERED
    language: Optional[str] = None
    language_confidence: float = 0.0
    rejection_reason: Optional[str] = None
    error_message: Optional[str] = None
    metadata_json: Optional[str] = None
    discovered_at: str = Field(default_factory=utc_now_iso)
    updated_at: str = Field(default_factory=utc_now_iso)


class CandidateRecord(BaseModel):
    """Database entity representing an extracted clip candidate."""
    id: Optional[int] = None
    candidate_id: str
    video_id: str
    start_sec: float
    end_sec: float
    duration_sec: float
    text: str = ""
    hook_score: float = 0.0
    payoff_score: float = 0.0
    self_contained_score: float = 0.0
    overall_score: float = 0.0
    reason: str = ""
    selected: bool = False
    visual_approved: bool = False
    visual_notes: str = ""
    created_at: str = Field(default_factory=utc_now_iso)


class RenderRecord(BaseModel):
    """Database entity representing a rendered short video."""
    id: Optional[int] = None
    candidate_id: Optional[int] = None
    video_id: str
    rendered_path: str
    duration_sec: float
    width: int = 1080
    height: int = 1920
    edit_plan_json: Optional[str] = None
    status: str = "rendering"  # rendering, rendered, render_failed
    qc_passed: bool = False
    qc_report_json: Optional[str] = None
    error_message: Optional[str] = None
    created_at: str = Field(default_factory=utc_now_iso)
    updated_at: str = Field(default_factory=utc_now_iso)


class UploadRecord(BaseModel):
    """Database entity representing a platform upload."""
    id: Optional[int] = None
    render_id: Optional[int] = None
    video_id: str
    platform: str = "youtube"
    status: str = "pending"  # uploading, completed, upload_failed, dry_run
    platform_video_id: Optional[str] = None
    url: Optional[str] = None
    title: str = ""
    description: str = ""
    dry_run: bool = False
    error_message: Optional[str] = None
    created_at: str = Field(default_factory=utc_now_iso)


# ==============================================================================
# Storage Repository
# ==============================================================================

class StorageRepository:
    """Thread-safe SQLite repository managing videos, candidates, renders, and uploads."""

    def __init__(self, db_path: Optional[Union[str, Path]] = None):
        self.db_path = Path(db_path or DEFAULT_DB_PATH).resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Opens a connection with WAL mode and foreign keys enabled."""
        conn = sqlite3.connect(str(self.db_path), timeout=20.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
        conn.execute("PRAGMA busy_timeout=5000;")
        return conn

    def init_db(self) -> None:
        """Initializes tables and indexes adhering to Blueprint Bab 18 schema."""
        with self._get_connection() as conn:
            # 1. Videos table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS videos (
                    video_id TEXT PRIMARY KEY,
                    url TEXT NOT NULL,
                    title TEXT DEFAULT '',
                    channel_title TEXT DEFAULT '',
                    duration_sec REAL DEFAULT 0.0,
                    published_at TEXT DEFAULT '',
                    status TEXT NOT NULL,
                    language TEXT,
                    language_confidence REAL DEFAULT 0.0,
                    rejection_reason TEXT,
                    error_message TEXT,
                    metadata_json TEXT,
                    discovered_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
            """)

            # Ensure columns exist in case of legacy schema
            existing_cols = {
                row["name"] for row in conn.execute("PRAGMA table_info(videos);").fetchall()
            }
            if "language" not in existing_cols:
                conn.execute("ALTER TABLE videos ADD COLUMN language TEXT;")
            if "language_confidence" not in existing_cols:
                conn.execute("ALTER TABLE videos ADD COLUMN language_confidence REAL DEFAULT 0.0;")
            if "rejection_reason" not in existing_cols:
                conn.execute("ALTER TABLE videos ADD COLUMN rejection_reason TEXT;")
            if "metadata_json" not in existing_cols:
                conn.execute("ALTER TABLE videos ADD COLUMN metadata_json TEXT;")
            if "discovered_at" not in existing_cols:
                conn.execute("ALTER TABLE videos ADD COLUMN discovered_at TEXT DEFAULT '';")
            if "updated_at" not in existing_cols:
                conn.execute("ALTER TABLE videos ADD COLUMN updated_at TEXT DEFAULT '';")

            # 2. Candidates table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS candidates (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    candidate_id TEXT NOT NULL,
                    video_id TEXT NOT NULL,
                    start_sec REAL NOT NULL,
                    end_sec REAL NOT NULL,
                    duration_sec REAL NOT NULL,
                    text TEXT DEFAULT '',
                    hook_score REAL DEFAULT 0.0,
                    payoff_score REAL DEFAULT 0.0,
                    self_contained_score REAL DEFAULT 0.0,
                    overall_score REAL DEFAULT 0.0,
                    reason TEXT DEFAULT '',
                    selected INTEGER DEFAULT 0,
                    visual_approved INTEGER DEFAULT 0,
                    visual_notes TEXT DEFAULT '',
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (video_id) REFERENCES videos(video_id) ON DELETE CASCADE
                );
            """)

            # 3. Renders table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS renders (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    candidate_id INTEGER,
                    video_id TEXT NOT NULL,
                    rendered_path TEXT NOT NULL,
                    duration_sec REAL NOT NULL,
                    width INTEGER DEFAULT 1080,
                    height INTEGER DEFAULT 1920,
                    edit_plan_json TEXT,
                    status TEXT NOT NULL,
                    qc_passed INTEGER DEFAULT 0,
                    qc_report_json TEXT,
                    error_message TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY (candidate_id) REFERENCES candidates(id) ON DELETE SET NULL,
                    FOREIGN KEY (video_id) REFERENCES videos(video_id) ON DELETE CASCADE
                );
            """)

            # 4. Uploads table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS uploads (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    render_id INTEGER,
                    video_id TEXT NOT NULL,
                    platform TEXT NOT NULL,
                    status TEXT NOT NULL,
                    platform_video_id TEXT,
                    url TEXT,
                    title TEXT DEFAULT '',
                    description TEXT DEFAULT '',
                    dry_run INTEGER DEFAULT 0,
                    error_message TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (render_id) REFERENCES renders(id) ON DELETE SET NULL,
                    FOREIGN KEY (video_id) REFERENCES videos(video_id) ON DELETE CASCADE
                );
            """)

            # Indexes for query performance
            conn.execute("CREATE INDEX IF NOT EXISTS idx_videos_status ON videos(status);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_candidates_video_id ON candidates(video_id);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_renders_video_id ON renders(video_id);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_uploads_video_id ON uploads(video_id);")
            conn.commit()

    # ==========================================================================
    # Video Operations
    # ==========================================================================

    def save_video(self, video: VideoRecord) -> VideoRecord:
        """Inserts or replaces a video record."""
        now = utc_now_iso()
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO videos (
                    video_id, url, title, channel_title, duration_sec,
                    published_at, status, language, language_confidence,
                    rejection_reason, error_message, metadata_json,
                    discovered_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(video_id) DO UPDATE SET
                    url = excluded.url,
                    title = excluded.title,
                    channel_title = excluded.channel_title,
                    duration_sec = excluded.duration_sec,
                    published_at = excluded.published_at,
                    status = excluded.status,
                    language = excluded.language,
                    language_confidence = excluded.language_confidence,
                    rejection_reason = excluded.rejection_reason,
                    error_message = excluded.error_message,
                    metadata_json = excluded.metadata_json,
                    updated_at = excluded.updated_at;
                """,
                (
                    video.video_id,
                    video.url,
                    video.title,
                    video.channel_title,
                    video.duration_sec,
                    video.published_at,
                    video.status.value if isinstance(video.status, PipelineStatus) else str(video.status),
                    video.language,
                    video.language_confidence,
                    video.rejection_reason,
                    video.error_message,
                    video.metadata_json,
                    video.discovered_at or now,
                    now,
                ),
            )
            conn.commit()
        return self.get_video(video.video_id) or video

    def update_video_status(
        self,
        video_id: str,
        new_status: Union[str, PipelineStatus],
        rejection_reason: Optional[str] = None,
        error_message: Optional[str] = None,
        validate: bool = True,
    ) -> VideoRecord:
        """Transitions a video's status with state machine validation.

        When transitioning to a terminal state (any rejection state or completed),
        reason and UTC timestamp are explicitly captured.
        """
        current = self.get_video(video_id)
        if not current:
            raise ValueError(f"Video '{video_id}' not found in database.")

        target = parse_state(new_status)

        if validate:
            trans = record_transition(
                current_state=current.status,
                target_state=target,
                reason=rejection_reason,
                error_message=error_message,
            )
            now = trans.timestamp
            target = trans.to_state
            if trans.is_terminal:
                rejection_reason = trans.reason
        else:
            now = utc_now_iso()
            if is_terminal(target):
                if not rejection_reason:
                    if target == PipelineStatus.COMPLETED:
                        rejection_reason = "Completed successfully."
                    else:
                        rejection_reason = error_message or f"Terminated with status: {target.value}"

        with self._get_connection() as conn:
            conn.execute(
                """
                UPDATE videos
                SET status = ?,
                    rejection_reason = COALESCE(?, rejection_reason),
                    error_message = COALESCE(?, error_message),
                    updated_at = ?
                WHERE video_id = ?;
                """,
                (
                    target.value,
                    rejection_reason,
                    error_message,
                    now,
                    video_id,
                ),
            )
            conn.commit()

        updated = self.get_video(video_id)
        if not updated:
            raise RuntimeError(f"Failed to retrieve updated video '{video_id}'")
        return updated

    def delete_video(self, video_id: str) -> bool:
        """Deletes video and cascades to candidates, renders, uploads."""
        with self._get_connection() as conn:
            cursor = conn.execute("DELETE FROM videos WHERE video_id = ?;", (video_id,))
            conn.commit()
            return cursor.rowcount > 0

    def video_exists(self, video_id: str) -> bool:
        """Checks if video exists in database."""
        with self._get_connection() as conn:
            row = conn.execute("SELECT 1 FROM videos WHERE video_id = ?;", (video_id,)).fetchone()
            return row is not None

    def get_video(self, video_id: str) -> Optional[VideoRecord]:
        """Fetches video by ID."""
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM videos WHERE video_id = ?;",
                (video_id,),
            ).fetchone()
            if not row:
                return None
            return VideoRecord(
                video_id=row["video_id"],
                url=row["url"],
                title=row["title"] or "",
                channel_title=row["channel_title"] or "",
                duration_sec=float(row["duration_sec"] or 0.0),
                published_at=row["published_at"] or "",
                status=parse_state(row["status"]),
                language=row["language"],
                language_confidence=float(row["language_confidence"] or 0.0),
                rejection_reason=row["rejection_reason"],
                error_message=row["error_message"],
                metadata_json=row["metadata_json"],
                discovered_at=row["discovered_at"] or "",
                updated_at=row["updated_at"] or "",
            )

    def list_videos(
        self,
        status: Optional[Union[str, PipelineStatus]] = None,
        limit: int = 100,
    ) -> List[VideoRecord]:
        """Lists videos with optional status filter."""
        with self._get_connection() as conn:
            if status is not None:
                st = parse_state(status).value
                rows = conn.execute(
                    "SELECT * FROM videos WHERE status = ? ORDER BY discovered_at DESC LIMIT ?;",
                    (st, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM videos ORDER BY discovered_at DESC LIMIT ?;",
                    (limit,),
                ).fetchall()

            return [
                VideoRecord(
                    video_id=r["video_id"],
                    url=r["url"],
                    title=r["title"] or "",
                    channel_title=r["channel_title"] or "",
                    duration_sec=float(r["duration_sec"] or 0.0),
                    published_at=r["published_at"] or "",
                    status=parse_state(r["status"]),
                    language=r["language"],
                    language_confidence=float(r["language_confidence"] or 0.0),
                    rejection_reason=r["rejection_reason"],
                    error_message=r["error_message"],
                    metadata_json=r["metadata_json"],
                    discovered_at=r["discovered_at"] or "",
                    updated_at=r["updated_at"] or "",
                )
                for r in rows
            ]

    # ==========================================================================
    # Candidate Operations
    # ==========================================================================

    def save_candidate(self, candidate: CandidateRecord) -> int:
        """Inserts a candidate and returns its auto-increment ID."""
        now = utc_now_iso()
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO candidates (
                    candidate_id, video_id, start_sec, end_sec, duration_sec,
                    text, hook_score, payoff_score, self_contained_score,
                    overall_score, reason, selected, visual_approved,
                    visual_notes, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    candidate.candidate_id,
                    candidate.video_id,
                    candidate.start_sec,
                    candidate.end_sec,
                    candidate.duration_sec,
                    candidate.text,
                    candidate.hook_score,
                    candidate.payoff_score,
                    candidate.self_contained_score,
                    candidate.overall_score,
                    candidate.reason,
                    1 if candidate.selected else 0,
                    1 if candidate.visual_approved else 0,
                    candidate.visual_notes,
                    candidate.created_at or now,
                ),
            )
            conn.commit()
            return int(cursor.lastrowid or 0)

    def get_candidate(self, candidate_pk: int) -> Optional[CandidateRecord]:
        """Fetches candidate by primary key."""
        with self._get_connection() as conn:
            row = conn.execute("SELECT * FROM candidates WHERE id = ?;", (candidate_pk,)).fetchone()
            if not row:
                return None
            return CandidateRecord(
                id=row["id"],
                candidate_id=row["candidate_id"],
                video_id=row["video_id"],
                start_sec=float(row["start_sec"]),
                end_sec=float(row["end_sec"]),
                duration_sec=float(row["duration_sec"]),
                text=row["text"] or "",
                hook_score=float(row["hook_score"] or 0.0),
                payoff_score=float(row["payoff_score"] or 0.0),
                self_contained_score=float(row["self_contained_score"] or 0.0),
                overall_score=float(row["overall_score"] or 0.0),
                reason=row["reason"] or "",
                selected=bool(row["selected"]),
                visual_approved=bool(row["visual_approved"]),
                visual_notes=row["visual_notes"] or "",
                created_at=row["created_at"] or "",
            )

    def get_candidates_for_video(self, video_id: str) -> List[CandidateRecord]:
        """Lists all candidates for a specific video."""
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM candidates WHERE video_id = ? ORDER BY overall_score DESC;",
                (video_id,),
            ).fetchall()
            return [
                CandidateRecord(
                    id=r["id"],
                    candidate_id=r["candidate_id"],
                    video_id=r["video_id"],
                    start_sec=float(r["start_sec"]),
                    end_sec=float(r["end_sec"]),
                    duration_sec=float(r["duration_sec"]),
                    text=r["text"] or "",
                    hook_score=float(r["hook_score"] or 0.0),
                    payoff_score=float(r["payoff_score"] or 0.0),
                    self_contained_score=float(r["self_contained_score"] or 0.0),
                    overall_score=float(r["overall_score"] or 0.0),
                    reason=r["reason"] or "",
                    selected=bool(r["selected"]),
                    visual_approved=bool(r["visual_approved"]),
                    visual_notes=r["visual_notes"] or "",
                    created_at=r["created_at"] or "",
                )
                for r in rows
            ]

    def mark_candidate_selected(
        self,
        candidate_pk: int,
        visual_approved: bool = False,
        visual_notes: str = "",
    ) -> None:
        """Marks candidate as selected and records visual verification notes."""
        with self._get_connection() as conn:
            conn.execute(
                """
                UPDATE candidates
                SET selected = 1,
                    visual_approved = ?,
                    visual_notes = ?
                WHERE id = ?;
                """,
                (1 if visual_approved else 0, visual_notes, candidate_pk),
            )
            conn.commit()

    def get_candidate_by_candidate_id(self, candidate_id: str) -> Optional[CandidateRecord]:
        """Fetches candidate by candidate_id string."""
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM candidates WHERE candidate_id = ? LIMIT 1;",
                (candidate_id,),
            ).fetchone()
            if not row:
                return None
            return CandidateRecord(
                id=row["id"],
                candidate_id=row["candidate_id"],
                video_id=row["video_id"],
                start_sec=float(row["start_sec"]),
                end_sec=float(row["end_sec"]),
                duration_sec=float(row["duration_sec"]),
                text=row["text"] or "",
                hook_score=float(row["hook_score"] or 0.0),
                payoff_score=float(row["payoff_score"] or 0.0),
                self_contained_score=float(row["self_contained_score"] or 0.0),
                overall_score=float(row["overall_score"] or 0.0),
                reason=row["reason"] or "",
                selected=bool(row["selected"]),
                visual_approved=bool(row["visual_approved"]),
                visual_notes=row["visual_notes"] or "",
                created_at=row["created_at"] or "",
            )

    def update_candidate(self, candidate: CandidateRecord) -> bool:
        """Updates an existing candidate record."""
        if candidate.id is None:
            raise ValueError("Candidate primary key (id) is required for update.")
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                UPDATE candidates
                SET candidate_id = ?,
                    video_id = ?,
                    start_sec = ?,
                    end_sec = ?,
                    duration_sec = ?,
                    text = ?,
                    hook_score = ?,
                    payoff_score = ?,
                    self_contained_score = ?,
                    overall_score = ?,
                    reason = ?,
                    selected = ?,
                    visual_approved = ?,
                    visual_notes = ?
                WHERE id = ?;
                """,
                (
                    candidate.candidate_id,
                    candidate.video_id,
                    candidate.start_sec,
                    candidate.end_sec,
                    candidate.duration_sec,
                    candidate.text,
                    candidate.hook_score,
                    candidate.payoff_score,
                    candidate.self_contained_score,
                    candidate.overall_score,
                    candidate.reason,
                    1 if candidate.selected else 0,
                    1 if candidate.visual_approved else 0,
                    candidate.visual_notes,
                    candidate.id,
                ),
            )
            conn.commit()
            return cursor.rowcount > 0

    def delete_candidate(self, candidate_pk: int) -> bool:
        """Deletes a candidate by primary key."""
        with self._get_connection() as conn:
            cursor = conn.execute("DELETE FROM candidates WHERE id = ?;", (candidate_pk,))
            conn.commit()
            return cursor.rowcount > 0

    # ==========================================================================
    # Render Operations
    # ==========================================================================

    def save_render(self, render: RenderRecord) -> int:
        """Inserts a render record and returns its ID."""
        now = utc_now_iso()
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO renders (
                    candidate_id, video_id, rendered_path, duration_sec,
                    width, height, edit_plan_json, status, qc_passed,
                    qc_report_json, error_message, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    render.candidate_id,
                    render.video_id,
                    render.rendered_path,
                    render.duration_sec,
                    render.width,
                    render.height,
                    render.edit_plan_json,
                    render.status,
                    1 if render.qc_passed else 0,
                    render.qc_report_json,
                    render.error_message,
                    render.created_at or now,
                    now,
                ),
            )
            conn.commit()
            return int(cursor.lastrowid or 0)

    def update_render_status(
        self,
        render_id: int,
        status: str,
        qc_passed: bool = False,
        qc_report_json: Optional[str] = None,
        error_message: Optional[str] = None,
    ) -> None:
        """Updates status and QC outcomes for a render."""
        now = utc_now_iso()
        with self._get_connection() as conn:
            conn.execute(
                """
                UPDATE renders
                SET status = ?,
                    qc_passed = ?,
                    qc_report_json = COALESCE(?, qc_report_json),
                    error_message = COALESCE(?, error_message),
                    updated_at = ?
                WHERE id = ?;
                """,
                (
                    status,
                    1 if qc_passed else 0,
                    qc_report_json,
                    error_message,
                    now,
                    render_id,
                ),
            )
            conn.commit()

    def get_render(self, render_id: int) -> Optional[RenderRecord]:
        """Fetches render by ID."""
        with self._get_connection() as conn:
            row = conn.execute("SELECT * FROM renders WHERE id = ?;", (render_id,)).fetchone()
            if not row:
                return None
            return RenderRecord(
                id=row["id"],
                candidate_id=row["candidate_id"],
                video_id=row["video_id"],
                rendered_path=row["rendered_path"],
                duration_sec=float(row["duration_sec"]),
                width=row["width"],
                height=row["height"],
                edit_plan_json=row["edit_plan_json"],
                status=row["status"],
                qc_passed=bool(row["qc_passed"]),
                qc_report_json=row["qc_report_json"],
                error_message=row["error_message"],
                created_at=row["created_at"] or "",
                updated_at=row["updated_at"] or "",
            )

    def get_renders_for_video(self, video_id: str) -> List[RenderRecord]:
        """Lists renders for a video."""
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM renders WHERE video_id = ? ORDER BY id DESC;",
                (video_id,),
            ).fetchall()
            return [
                RenderRecord(
                    id=r["id"],
                    candidate_id=r["candidate_id"],
                    video_id=r["video_id"],
                    rendered_path=r["rendered_path"],
                    duration_sec=float(r["duration_sec"]),
                    width=r["width"],
                    height=r["height"],
                    edit_plan_json=r["edit_plan_json"],
                    status=r["status"],
                    qc_passed=bool(r["qc_passed"]),
                    qc_report_json=r["qc_report_json"],
                    error_message=r["error_message"],
                    created_at=r["created_at"] or "",
                    updated_at=r["updated_at"] or "",
                )
                for r in rows
            ]

    def delete_render(self, render_id: int) -> bool:
        """Deletes a render by ID."""
        with self._get_connection() as conn:
            cursor = conn.execute("DELETE FROM renders WHERE id = ?;", (render_id,))
            conn.commit()
            return cursor.rowcount > 0

    # ==========================================================================
    # Upload Operations
    # ==========================================================================

    def save_upload(self, upload: UploadRecord) -> int:
        """Inserts an upload record."""
        now = utc_now_iso()
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO uploads (
                    render_id, video_id, platform, status, platform_video_id,
                    url, title, description, dry_run, error_message, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    upload.render_id,
                    upload.video_id,
                    upload.platform,
                    upload.status,
                    upload.platform_video_id,
                    upload.url,
                    upload.title,
                    upload.description,
                    1 if upload.dry_run else 0,
                    upload.error_message,
                    upload.created_at or now,
                ),
            )
            conn.commit()
            return int(cursor.lastrowid or 0)

    def update_upload_status(
        self,
        upload_id: int,
        status: str,
        platform_video_id: Optional[str] = None,
        url: Optional[str] = None,
        error_message: Optional[str] = None,
    ) -> None:
        """Updates upload status and final published URL."""
        with self._get_connection() as conn:
            conn.execute(
                """
                UPDATE uploads
                SET status = ?,
                    platform_video_id = COALESCE(?, platform_video_id),
                    url = COALESCE(?, url),
                    error_message = COALESCE(?, error_message)
                WHERE id = ?;
                """,
                (status, platform_video_id, url, error_message, upload_id),
            )
            conn.commit()

    def get_uploads_for_video(self, video_id: str) -> List[UploadRecord]:
        """Lists uploads for a video."""
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM uploads WHERE video_id = ? ORDER BY id DESC;",
                (video_id,),
            ).fetchall()
            return [
                UploadRecord(
                    id=r["id"],
                    render_id=r["render_id"],
                    video_id=r["video_id"],
                    platform=r["platform"],
                    status=r["status"],
                    platform_video_id=r["platform_video_id"],
                    url=r["url"],
                    title=r["title"] or "",
                    description=r["description"] or "",
                    dry_run=bool(r["dry_run"]),
                    error_message=r["error_message"],
                    created_at=r["created_at"] or "",
                )
                for r in rows
            ]

    def get_upload(self, upload_id: int) -> Optional[UploadRecord]:
        """Fetches an upload by primary key."""
        with self._get_connection() as conn:
            row = conn.execute("SELECT * FROM uploads WHERE id = ?;", (upload_id,)).fetchone()
            if not row:
                return None
            return UploadRecord(
                id=row["id"],
                render_id=row["render_id"],
                video_id=row["video_id"],
                platform=row["platform"],
                status=row["status"],
                platform_video_id=row["platform_video_id"],
                url=row["url"],
                title=row["title"] or "",
                description=row["description"] or "",
                dry_run=bool(row["dry_run"]),
                error_message=row["error_message"],
                created_at=row["created_at"] or "",
            )

    def delete_upload(self, upload_id: int) -> bool:
        """Deletes an upload by primary key."""
        with self._get_connection() as conn:
            cursor = conn.execute("DELETE FROM uploads WHERE id = ?;", (upload_id,))
            conn.commit()
            return cursor.rowcount > 0

    def get_video_history(self, video_id: str) -> Optional[Dict[str, Any]]:
        """Returns the full relational lifecycle graph for a video."""
        video = self.get_video(video_id)
        if not video:
            return None

        candidates = self.get_candidates_for_video(video_id)
        renders = self.get_renders_for_video(video_id)
        uploads = self.get_uploads_for_video(video_id)

        return {
            "video": video.model_dump(),
            "candidates": [c.model_dump() for c in candidates],
            "renders": [r.model_dump() for r in renders],
            "uploads": [u.model_dump() for u in uploads],
        }
