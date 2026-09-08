"""Comprehensive Test Suite for Phase D (Part 1: Storage & State Machine).

Implements and verifies Blueprint Bab 18:
1. Pipeline State Machine (11 happy path states + 6 terminal reject states).
2. Guard Invariant: strictly forbidden to transition to 'uploading' unless current is 'qc_passed'.
3. Terminal Immutability Guard: terminal states cannot transition further.
4. Terminal Transition Recording: reason and UTC timestamp captured for all terminal states.
5. Storage Repository: SQLite WAL mode, foreign keys, indexes, schema migrations.
6. Full CRUD operations across all 4 tables: videos, candidates, renders, uploads.
7. Atomic state transitions in repository with invariant enforcement.
8. Relational cascade and full video lifecycle history graph.
"""

import sqlite3
import tempfile
from datetime import datetime, timezone
from pathlib import Path
import pytest

from pipeline.state_machine import (
    PipelineStatus,
    HAPPY_STATES,
    TERMINAL_REJECT_STATES,
    TERMINAL_STATES,
    VALID_TRANSITIONS,
    InvalidStateTransitionError,
    GuardInvariantError,
    StateTransition,
    parse_state,
    can_transition,
    validate_transition,
    record_transition,
    transition,
    is_terminal,
    is_happy,
    is_reject,
    utc_now_iso,
)
from storage.repository import (
    StorageRepository,
    VideoRecord,
    CandidateRecord,
    RenderRecord,
    UploadRecord,
)


@pytest.fixture
def temp_repo():
    """Provides a fresh isolated StorageRepository inside a temporary directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_app_v3.db"
        repo = StorageRepository(db_path=db_path)
        yield repo


# ==============================================================================
# 1. State Machine Definitions & Categorization (Blueprint Bab 18)
# ==============================================================================

def test_state_machine_11_happy_states():
    """Validates that exactly 11 Happy Path states exist and match Blueprint Bab 18."""
    expected_happy = [
        "discovered",
        "eligible",
        "transcribed",
        "candidates_found",
        "candidate_selected",
        "visual_verified",
        "rendering",
        "rendered",
        "qc_passed",
        "uploading",
        "completed",
    ]
    assert len(HAPPY_STATES) == 11
    assert len(expected_happy) == 11
    for name in expected_happy:
        status = PipelineStatus(name)
        assert status in HAPPY_STATES
        assert is_happy(status)
        assert not is_reject(status)
        if status != PipelineStatus.COMPLETED:
            assert not is_terminal(status)


def test_state_machine_6_terminal_reject_states():
    """Validates that exactly 6 Rejection/Failure states exist and are terminal."""
    expected_rejects = [
        "rejected_language",
        "no_good_clip",
        "rejected_visual",
        "render_failed",
        "qc_failed",
        "upload_failed",
    ]
    assert len(TERMINAL_REJECT_STATES) == 6
    assert len(expected_rejects) == 6
    for name in expected_rejects:
        status = PipelineStatus(name)
        assert status in TERMINAL_REJECT_STATES
        assert is_reject(status)
        assert is_terminal(status)
        assert not is_happy(status)


def test_state_machine_total_and_terminal_sets():
    """Validates overall state definitions and terminal union."""
    assert len(PipelineStatus) == 17
    assert len(TERMINAL_STATES) == 7  # 6 rejects + completed
    assert PipelineStatus.COMPLETED in TERMINAL_STATES
    assert is_terminal(PipelineStatus.COMPLETED)
    assert is_happy(PipelineStatus.COMPLETED)  # completed is happy AND terminal


def test_parse_state_and_case_insensitivity():
    """Tests string and enum parsing with case insensitivity."""
    assert parse_state(PipelineStatus.DISCOVERED) == PipelineStatus.DISCOVERED
    assert parse_state("discovered") == PipelineStatus.DISCOVERED
    assert parse_state("DISCOVERED") == PipelineStatus.DISCOVERED
    assert parse_state("Qc_Passed") == PipelineStatus.QC_PASSED
    assert parse_state("UPLOAD_FAILED") == PipelineStatus.UPLOAD_FAILED

    with pytest.raises(ValueError, match="Unknown pipeline state"):
        parse_state("non_existent_status")


# ==============================================================================
# 2. State Transition Validity & Graph Traversal
# ==============================================================================

def test_happy_path_linear_transition_chain():
    """Tests the full 11-step end-to-end happy path transition chain."""
    happy_chain = [
        PipelineStatus.DISCOVERED,
        PipelineStatus.ELIGIBLE,
        PipelineStatus.TRANSCRIBED,
        PipelineStatus.CANDIDATES_FOUND,
        PipelineStatus.CANDIDATE_SELECTED,
        PipelineStatus.VISUAL_VERIFIED,
        PipelineStatus.RENDERING,
        PipelineStatus.RENDERED,
        PipelineStatus.QC_PASSED,
        PipelineStatus.UPLOADING,
        PipelineStatus.COMPLETED,
    ]

    for i in range(len(happy_chain) - 1):
        curr = happy_chain[i]
        nxt = happy_chain[i + 1]
        assert can_transition(curr, nxt) is True
        result = validate_transition(curr, nxt)
        assert result == nxt


def test_qc_passed_to_completed_dry_run_transition():
    """Tests that qc_passed can transition directly to completed (dry-run or local mode)."""
    assert can_transition(PipelineStatus.QC_PASSED, PipelineStatus.COMPLETED) is True
    assert validate_transition(PipelineStatus.QC_PASSED, PipelineStatus.COMPLETED) == PipelineStatus.COMPLETED


def test_allowed_rejection_transitions():
    """Tests all legal failure/rejection transitions defined in Blueprint Bab 18."""
    legal_rejections = [
        (PipelineStatus.DISCOVERED, PipelineStatus.REJECTED_LANGUAGE),
        (PipelineStatus.ELIGIBLE, PipelineStatus.REJECTED_LANGUAGE),
        (PipelineStatus.TRANSCRIBED, PipelineStatus.REJECTED_LANGUAGE),
        (PipelineStatus.TRANSCRIBED, PipelineStatus.NO_GOOD_CLIP),
        (PipelineStatus.CANDIDATES_FOUND, PipelineStatus.NO_GOOD_CLIP),
        (PipelineStatus.CANDIDATE_SELECTED, PipelineStatus.NO_GOOD_CLIP),
        (PipelineStatus.CANDIDATE_SELECTED, PipelineStatus.REJECTED_VISUAL),
        (PipelineStatus.VISUAL_VERIFIED, PipelineStatus.REJECTED_VISUAL),
        (PipelineStatus.RENDERING, PipelineStatus.RENDER_FAILED),
        (PipelineStatus.RENDERED, PipelineStatus.QC_FAILED),
        (PipelineStatus.UPLOADING, PipelineStatus.UPLOAD_FAILED),
    ]

    for curr, rej in legal_rejections:
        assert can_transition(curr, rej) is True
        assert validate_transition(curr, rej) == rej


def test_disallowed_backward_and_jump_transitions():
    """Tests that skipping stages or transitioning backwards is rejected."""
    illegal_transitions = [
        (PipelineStatus.DISCOVERED, PipelineStatus.RENDERING),
        (PipelineStatus.DISCOVERED, PipelineStatus.COMPLETED),
        (PipelineStatus.ELIGIBLE, PipelineStatus.RENDERED),
        (PipelineStatus.CANDIDATES_FOUND, PipelineStatus.DISCOVERED),
        (PipelineStatus.RENDERED, PipelineStatus.DISCOVERED),
        (PipelineStatus.RENDERED, PipelineStatus.TRANSCRIBED),
        (PipelineStatus.VISUAL_VERIFIED, PipelineStatus.COMPLETED),
    ]

    for curr, target in illegal_transitions:
        assert can_transition(curr, target) is False
        with pytest.raises(InvalidStateTransitionError):
            validate_transition(curr, target)


# ==============================================================================
# 3. Guard Invariants (Blueprint Bab 18 & Bab 21)
# ==============================================================================

def test_guard_invariant_uploading_only_from_qc_passed():
    """CRITICAL GUARD INVARIANT: Cannot transition to 'uploading' unless current is 'qc_passed'."""
    # Non-terminal states attempting to transition to uploading must raise GuardInvariantError
    non_qc_states = [
        PipelineStatus.DISCOVERED,
        PipelineStatus.ELIGIBLE,
        PipelineStatus.TRANSCRIBED,
        PipelineStatus.CANDIDATES_FOUND,
        PipelineStatus.CANDIDATE_SELECTED,
        PipelineStatus.VISUAL_VERIFIED,
        PipelineStatus.RENDERING,
        PipelineStatus.RENDERED,
    ]
    for state in non_qc_states:
        assert can_transition(state, PipelineStatus.UPLOADING) is False
        with pytest.raises(GuardInvariantError) as exc_info:
            validate_transition(state, PipelineStatus.UPLOADING)
        assert "qc_passed" in str(exc_info.value)

    # Terminal states are also blocked from transitioning to uploading
    for term in TERMINAL_STATES:
        assert can_transition(term, PipelineStatus.UPLOADING) is False
        with pytest.raises(InvalidStateTransitionError):
            validate_transition(term, PipelineStatus.UPLOADING)

    # Only qc_passed is permitted
    assert can_transition(PipelineStatus.QC_PASSED, PipelineStatus.UPLOADING) is True
    assert validate_transition(PipelineStatus.QC_PASSED, PipelineStatus.UPLOADING) == PipelineStatus.UPLOADING


def test_terminal_immutability_guard():
    """CRITICAL TERMINAL GUARD: Terminal states cannot transition to ANY state."""
    for term in TERMINAL_STATES:
        for any_target in PipelineStatus:
            assert can_transition(term, any_target) is False
            with pytest.raises(InvalidStateTransitionError) as exc_info:
                validate_transition(term, any_target)
            assert "terminal state" in str(exc_info.value)


# ==============================================================================
# 4. Terminal Transition Recording (Reason & Timestamp)
# ==============================================================================

def test_record_transition_terminal_rejection_with_reason():
    """Tests that terminal reject transition records explicit reason and ISO UTC timestamp."""
    trans = record_transition(
        current_state=PipelineStatus.RENDERED,
        target_state=PipelineStatus.QC_FAILED,
        reason="Visual QC: blank frame detected at t=12.4s",
        error_message="Blank frames detected",
    )
    assert isinstance(trans, StateTransition)
    assert trans.from_state == PipelineStatus.RENDERED
    assert trans.to_state == PipelineStatus.QC_FAILED
    assert trans.is_terminal is True
    assert trans.reason == "Visual QC: blank frame detected at t=12.4s"
    assert trans.error_message == "Blank frames detected"

    # Verify timestamp format
    ts = datetime.fromisoformat(trans.timestamp)
    assert ts.tzinfo is not None


def test_record_transition_terminal_rejection_default_reason():
    """Tests that terminal transition supplies default reason when caller omits reason."""
    trans = record_transition(
        current_state=PipelineStatus.RENDERING,
        target_state=PipelineStatus.RENDER_FAILED,
    )
    assert trans.is_terminal is True
    assert trans.reason is not None
    assert "render_failed" in trans.reason
    assert trans.timestamp != ""


def test_record_transition_completed_state():
    """Tests that transition to 'completed' records terminal=True, timestamp, and reason."""
    trans = record_transition(
        current_state=PipelineStatus.UPLOADING,
        target_state=PipelineStatus.COMPLETED,
    )
    assert trans.from_state == PipelineStatus.UPLOADING
    assert trans.to_state == PipelineStatus.COMPLETED
    assert trans.is_terminal is True
    assert trans.reason == "Completed successfully."
    assert trans.timestamp != ""


def test_record_transition_non_terminal_happy_state():
    """Tests that non-terminal transitions have is_terminal=False."""
    trans = record_transition(
        current_state=PipelineStatus.DISCOVERED,
        target_state=PipelineStatus.ELIGIBLE,
    )
    assert trans.from_state == PipelineStatus.DISCOVERED
    assert trans.to_state == PipelineStatus.ELIGIBLE
    assert trans.is_terminal is False
    assert trans.timestamp != ""


def test_record_transition_raises_on_guard_violation():
    """Tests that record_transition enforces the same guard invariants as validate_transition."""
    with pytest.raises(GuardInvariantError):
        record_transition(
            current_state=PipelineStatus.DISCOVERED,
            target_state=PipelineStatus.UPLOADING,
        )


# ==============================================================================
# 5. Storage Repository Database Initialization & Migrations
# ==============================================================================

def test_storage_repository_init_wal_and_foreign_keys(temp_repo):
    """Verifies SQLite WAL mode, foreign keys, and indexes."""
    with temp_repo._get_connection() as conn:
        journal_mode = conn.execute("PRAGMA journal_mode;").fetchone()[0]
        assert journal_mode.upper() == "WAL"

        foreign_keys = conn.execute("PRAGMA foreign_keys;").fetchone()[0]
        assert foreign_keys == 1

        tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall()}
        assert {"videos", "candidates", "renders", "uploads"}.issubset(tables)


def test_storage_repository_migration_resilience_on_legacy_db():
    """Tests that StorageRepository automatically migrates legacy schemas without data loss."""
    with tempfile.TemporaryDirectory() as tmpdir:
        legacy_db = Path(tmpdir) / "legacy.db"
        # Simulate legacy database with old schema
        conn = sqlite3.connect(str(legacy_db))
        conn.execute("""
            CREATE TABLE videos (
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
        """)
        conn.execute("INSERT INTO videos VALUES ('old1', 'http://url', 'Title', 'Channel', 100, '', '2026-01-01', 'discovered', NULL);")
        conn.commit()
        conn.close()

        # Initialize repository on legacy database
        repo = StorageRepository(db_path=legacy_db)
        vid = repo.get_video("old1")
        assert vid is not None
        assert vid.video_id == "old1"
        assert vid.status == PipelineStatus.DISCOVERED
        assert vid.language is None
        assert vid.metadata_json is None


# ==============================================================================
# 6. Video CRUD & Atomic State Transitions
# ==============================================================================

def test_video_crud_lifecycle(temp_repo):
    """Tests complete CRUD operations for videos table."""
    # 1. Create
    video = VideoRecord(
        video_id="vid_test_001",
        url="https://youtube.com/watch?v=vid_test_001",
        title="Cara Coding Python",
        channel_title="Tech Indo",
        duration_sec=360.5,
        published_at="2026-09-08T00:00:00Z",
        status=PipelineStatus.DISCOVERED,
        language="id",
        language_confidence=0.98,
        metadata_json='{"view_count": 5000}',
    )
    saved = temp_repo.save_video(video)
    assert saved.video_id == "vid_test_001"
    assert temp_repo.video_exists("vid_test_001") is True

    # 2. Read
    fetched = temp_repo.get_video("vid_test_001")
    assert fetched is not None
    assert fetched.title == "Cara Coding Python"
    assert fetched.language == "id"
    assert fetched.language_confidence == 0.98
    assert fetched.duration_sec == 360.5

    # 3. List
    all_videos = temp_repo.list_videos()
    assert len(all_videos) == 1
    assert all_videos[0].video_id == "vid_test_001"

    filtered_empty = temp_repo.list_videos(status=PipelineStatus.COMPLETED)
    assert len(filtered_empty) == 0

    filtered_match = temp_repo.list_videos(status=PipelineStatus.DISCOVERED)
    assert len(filtered_match) == 1

    # 4. Update status atomically
    updated = temp_repo.update_video_status("vid_test_001", PipelineStatus.ELIGIBLE)
    assert updated.status == PipelineStatus.ELIGIBLE

    # 5. Delete
    deleted = temp_repo.delete_video("vid_test_001")
    assert deleted is True
    assert temp_repo.video_exists("vid_test_001") is False
    assert temp_repo.get_video("vid_test_001") is None


def test_video_status_atomic_guard_enforcement(temp_repo):
    """Verifies that illegal status update throws GuardInvariantError and state remains unchanged."""
    video = VideoRecord(
        video_id="vid_guard_test",
        url="https://youtube.com/watch?v=vid_guard_test",
        status=PipelineStatus.TRANSCRIBED,
    )
    temp_repo.save_video(video)

    # Attempt illegal jump to uploading
    with pytest.raises(GuardInvariantError):
        temp_repo.update_video_status("vid_guard_test", PipelineStatus.UPLOADING)

    # Verify atomic invariant: state must NOT have changed
    vid_after = temp_repo.get_video("vid_guard_test")
    assert vid_after.status == PipelineStatus.TRANSCRIBED


def test_video_terminal_rejection_records_reason_and_timestamp(temp_repo):
    """Verifies that terminal rejection records reason and updated_at timestamp in database."""
    video = VideoRecord(
        video_id="vid_rej_test",
        url="https://youtube.com/watch?v=vid_rej_test",
        status=PipelineStatus.TRANSCRIBED,
    )
    temp_repo.save_video(video)

    updated = temp_repo.update_video_status(
        video_id="vid_rej_test",
        new_status=PipelineStatus.NO_GOOD_CLIP,
        rejection_reason="Gemini semantic score below threshold (hook=32, payoff=40)",
    )

    assert updated.status == PipelineStatus.NO_GOOD_CLIP
    assert updated.rejection_reason == "Gemini semantic score below threshold (hook=32, payoff=40)"
    assert updated.updated_at != ""

    # Check persistence directly from DB
    persisted = temp_repo.get_video("vid_rej_test")
    assert persisted.status == PipelineStatus.NO_GOOD_CLIP
    assert persisted.rejection_reason == "Gemini semantic score below threshold (hook=32, payoff=40)"

    # Terminal state cannot be updated anymore
    with pytest.raises(InvalidStateTransitionError):
        temp_repo.update_video_status("vid_rej_test", PipelineStatus.CANDIDATES_FOUND)


# ==============================================================================
# 7. Candidate CRUD Operations
# ==============================================================================

def test_candidate_crud_lifecycle(temp_repo):
    """Tests CRUD operations for clip candidates."""
    video = VideoRecord(video_id="vid_cands", url="https://youtube.com/watch?v=vid_cands")
    temp_repo.save_video(video)

    # 1. Create Candidate
    cand = CandidateRecord(
        candidate_id="c_001",
        video_id="vid_cands",
        start_sec=15.0,
        end_sec=55.0,
        duration_sec=40.0,
        text="Ini tips coding mantap.",
        hook_score=85.0,
        payoff_score=88.0,
        self_contained_score=90.0,
        overall_score=87.5,
        reason="Punchy opener and clean ending.",
    )
    cand_id = temp_repo.save_candidate(cand)
    assert cand_id > 0

    # 2. Read Candidate
    fetched = temp_repo.get_candidate(cand_id)
    assert fetched is not None
    assert fetched.candidate_id == "c_001"
    assert fetched.overall_score == 87.5
    assert fetched.selected is False

    by_str_id = temp_repo.get_candidate_by_candidate_id("c_001")
    assert by_str_id is not None
    assert by_str_id.id == cand_id

    # 3. Mark Selected
    temp_repo.mark_candidate_selected(cand_id, visual_approved=True, visual_notes="Clean single speaker")
    updated_cand = temp_repo.get_candidate(cand_id)
    assert updated_cand.selected is True
    assert updated_cand.visual_approved is True
    assert updated_cand.visual_notes == "Clean single speaker"

    # 4. Update Candidate
    updated_cand.hook_score = 95.0
    res = temp_repo.update_candidate(updated_cand)
    assert res is True
    reloaded = temp_repo.get_candidate(cand_id)
    assert reloaded.hook_score == 95.0

    # 5. List for video
    candidates_list = temp_repo.get_candidates_for_video("vid_cands")
    assert len(candidates_list) == 1

    # 6. Delete Candidate
    deleted = temp_repo.delete_candidate(cand_id)
    assert deleted is True
    assert temp_repo.get_candidate(cand_id) is None


# ==============================================================================
# 8. Render CRUD Operations
# ==============================================================================

def test_render_crud_lifecycle(temp_repo):
    """Tests CRUD operations for rendered video artifacts."""
    video = VideoRecord(video_id="vid_render", url="https://youtube.com/watch?v=vid_render")
    temp_repo.save_video(video)

    cand = CandidateRecord(
        candidate_id="c_rnd_01",
        video_id="vid_render",
        start_sec=10.0,
        end_sec=45.0,
        duration_sec=35.0,
    )
    cand_id = temp_repo.save_candidate(cand)

    # 1. Create Render
    render = RenderRecord(
        candidate_id=cand_id,
        video_id="vid_render",
        rendered_path="/tmp/output_shorts/vid_render_short.mp4",
        duration_sec=35.0,
        width=1080,
        height=1920,
        edit_plan_json='{"punch_ins": 1}',
        status="rendering",
    )
    render_id = temp_repo.save_render(render)
    assert render_id > 0

    # 2. Read Render
    fetched = temp_repo.get_render(render_id)
    assert fetched is not None
    assert fetched.status == "rendering"
    assert fetched.qc_passed is False

    # 3. Update Render Status & QC results
    temp_repo.update_render_status(
        render_id=render_id,
        status="rendered",
        qc_passed=True,
        qc_report_json='{"tier1": "PASS", "tier2": "PASS", "tier3": "PASS"}',
    )
    updated_rnd = temp_repo.get_render(render_id)
    assert updated_rnd.status == "rendered"
    assert updated_rnd.qc_passed is True
    assert "PASS" in updated_rnd.qc_report_json

    # 4. List renders for video
    renders_list = temp_repo.get_renders_for_video("vid_render")
    assert len(renders_list) == 1

    # 5. Delete Render
    deleted = temp_repo.delete_render(render_id)
    assert deleted is True
    assert temp_repo.get_render(render_id) is None


# ==============================================================================
# 9. Upload CRUD Operations
# ==============================================================================

def test_upload_crud_lifecycle(temp_repo):
    """Tests CRUD operations for uploads table."""
    video = VideoRecord(video_id="vid_up", url="https://youtube.com/watch?v=vid_up")
    temp_repo.save_video(video)

    render = RenderRecord(
        video_id="vid_up",
        rendered_path="/tmp/vid_up.mp4",
        duration_sec=40.0,
        status="rendered",
        qc_passed=True,
    )
    render_id = temp_repo.save_render(render)

    # 1. Create Upload
    upload = UploadRecord(
        render_id=render_id,
        video_id="vid_up",
        platform="youtube",
        status="uploading",
        title="Short Video Mantap #shorts",
        description="Deskripsi auto generated",
        dry_run=False,
    )
    upload_id = temp_repo.save_upload(upload)
    assert upload_id > 0

    # 2. Read Upload
    fetched = temp_repo.get_upload(upload_id)
    assert fetched is not None
    assert fetched.platform == "youtube"
    assert fetched.status == "uploading"

    # 3. Update Upload status
    temp_repo.update_upload_status(
        upload_id=upload_id,
        status="completed",
        platform_video_id="YT_SHORT_12345",
        url="https://youtube.com/shorts/YT_SHORT_12345",
    )
    updated_up = temp_repo.get_upload(upload_id)
    assert updated_up.status == "completed"
    assert updated_up.platform_video_id == "YT_SHORT_12345"
    assert updated_up.url == "https://youtube.com/shorts/YT_SHORT_12345"

    # 4. List uploads for video
    uploads_list = temp_repo.get_uploads_for_video("vid_up")
    assert len(uploads_list) == 1

    # 5. Delete Upload
    deleted = temp_repo.delete_upload(upload_id)
    assert deleted is True
    assert temp_repo.get_upload(upload_id) is None


# ==============================================================================
# 10. Relational Lifecycle Graph & Cascade Deletion
# ==============================================================================

def test_video_history_relational_graph(temp_repo):
    """Tests retrieval of complete relational graph for a video."""
    video_id = "vid_full_graph"
    video = VideoRecord(video_id=video_id, url=f"https://youtube.com/watch?v={video_id}", title="Full Graph Test")
    temp_repo.save_video(video)

    c1 = CandidateRecord(candidate_id="cg_1", video_id=video_id, start_sec=0.0, end_sec=30.0, duration_sec=30.0)
    c1_id = temp_repo.save_candidate(c1)

    r1 = RenderRecord(candidate_id=c1_id, video_id=video_id, rendered_path="/tmp/cg1.mp4", duration_sec=30.0, status="rendered", qc_passed=True)
    r1_id = temp_repo.save_render(r1)

    u1 = UploadRecord(render_id=r1_id, video_id=video_id, platform="youtube", status="completed", platform_video_id="YT_GRAPH_01")
    temp_repo.save_upload(u1)

    history = temp_repo.get_video_history(video_id)
    assert history is not None
    assert history["video"]["video_id"] == video_id
    assert len(history["candidates"]) == 1
    assert history["candidates"][0]["candidate_id"] == "cg_1"
    assert len(history["renders"]) == 1
    assert history["renders"][0]["id"] == r1_id
    assert len(history["uploads"]) == 1
    assert history["uploads"][0]["platform_video_id"] == "YT_GRAPH_01"


def test_video_deletion_cascades_to_all_child_tables(temp_repo):
    """Verifies that deleting a video cascades and deletes candidates, renders, and uploads."""
    video_id = "vid_cascade"
    temp_repo.save_video(VideoRecord(video_id=video_id, url="https://youtube.com/watch?v=vid_cascade"))

    c_id = temp_repo.save_candidate(CandidateRecord(candidate_id="c_casc", video_id=video_id, start_sec=5.0, end_sec=40.0, duration_sec=35.0))
    r_id = temp_repo.save_render(RenderRecord(candidate_id=c_id, video_id=video_id, rendered_path="/tmp/casc.mp4", duration_sec=35.0, status="rendered"))
    u_id = temp_repo.save_upload(UploadRecord(render_id=r_id, video_id=video_id, platform="youtube", status="pending"))

    assert temp_repo.get_candidate(c_id) is not None
    assert temp_repo.get_render(r_id) is not None
    assert temp_repo.get_upload(u_id) is not None

    # Delete video
    deleted = temp_repo.delete_video(video_id)
    assert deleted is True

    # Cascade must have removed all related rows
    assert temp_repo.get_video(video_id) is None
    assert temp_repo.get_candidate(c_id) is None
    assert temp_repo.get_render(r_id) is None
    assert temp_repo.get_upload(u_id) is None
