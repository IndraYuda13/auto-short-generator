"""Comprehensive Unit & Integration Test Suite for Phase D.

Verifies:
1. Bab 18: Storage & Database (SQLite repository at data/app_v3.db with 11 happy & 6 terminal states)
2. Bab 21: State Machine & Transition Validator with strict guard invariants
3. Bab 17: Strict Upload Gate & YouTube Shorts Uploader (dry_run & live flows)
4. Bab 22: Full End-to-End Auto Clipper Orchestrator (happy path, rejection paths, discovery cycle)
"""

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch
import pytest

from pipeline.state_machine import (
    PipelineStatus,
    HAPPY_STATES,
    TERMINAL_REJECT_STATES,
    TERMINAL_STATES,
    VALID_TRANSITIONS,
    InvalidStateTransitionError,
    GuardInvariantError,
    parse_state,
    can_transition,
    validate_transition,
    is_terminal,
    is_happy,
    is_reject,
)
from storage.repository import (
    StorageRepository,
    VideoRecord,
    CandidateRecord,
    RenderRecord,
    UploadRecord,
)
from upload.uploader import (
    all_gates_pass,
    UploadGateCheck,
    UploadGateRejectedError,
    StrictUploadGate,
    YouTubeShortsUploader,
)
from pipeline.orchestrator import (
    AutoClipperOrchestrator,
    PipelineResult,
)
from discovery.searcher import VideoSourceMeta
from discovery.source_filter import EligibilityResult
from language.language_gate import LanguageGateResult
from transcription.transcript_provider import TranscriptSegment
from analysis.candidate_generator import CandidateWindow
from analysis.semantic_scorer import SemanticScore
from analysis.boundary_refiner import RefinementResult
from analysis.visual_analyzer import VisualAnalysisReport
from analysis.visual_director import VisualDirectorVerdict
from editing.framing import FramingDecision
from editing.edit_plan import EditPlan, SceneCrop
from editing.renderer import RenderResult
from quality import (
    ThreeTierQCReport,
    TechnicalQCResult,
    VisualQCResult,
    PerceptualQCResult,
)


# ==============================================================================
# Fixtures
# ==============================================================================

@pytest.fixture
def temp_db(tmp_path: Path) -> StorageRepository:
    """Provides a fresh isolated SQLite repository for testing."""
    db_file = tmp_path / "test_app_v3.db"
    return StorageRepository(db_path=db_file)


@pytest.fixture
def valid_gate_kwargs() -> Dict[str, Any]:
    """Default kwargs satisfying all 8 production gates."""
    return {
        "language_gate": True,
        "semantic_clip_gate": True,
        "visual_viability_gate": True,
        "boundary_gate": True,
        "render_success": True,
        "technical_qc": True,
        "visual_qc": True,
        "perceptual_qc": True,
    }


@pytest.fixture
def dummy_video_file(tmp_path: Path) -> Path:
    """Creates a dummy valid-sized video file (>100KB) for uploader and render tests."""
    v_path = tmp_path / "test_short_video.mp4"
    # Write 150KB of deterministic binary data
    v_path.write_bytes(b"\x00" * (150 * 1024))
    return v_path


@pytest.fixture(autouse=True)
def mock_gemini_remote_calls(monkeypatch):
    """Mocks 9router remote calls in pipeline unit tests for deterministic execution."""
    from analysis.visual_preflight import VisualPreflightResult
    from quality.gemini_video_qc import GeminiVideoQCResult

    def fake_preflight(self, video_path, transcript_excerpt=""):
        return VisualPreflightResult(
            usable=True,
            existing_visible_subtitles=False,
            shot_complexity="low",
            subject_composition="acceptable",
            recommended_layout="SAFE_WIDE",
            blocking_issues=[],
            notes="Mock preflight pass",
        )

    def fake_gemini_video_qc(self, video_path, transcript_text="", edit_plan=None):
        return GeminiVideoQCResult(
            passed=True,
            score=85,
            blocking_reasons=[],
            summary="Mock Gemini Video QC pass",
        )

    monkeypatch.setattr("analysis.visual_preflight.VisualPreflight.preflight_clip", fake_preflight)
    monkeypatch.setattr("quality.gemini_video_qc.GeminiNativeVideoQC.evaluate_video", fake_gemini_video_qc)


# ==============================================================================
# 1. Bab 21: State Machine & Transition Validator Tests
# ==============================================================================

def test_state_machine_happy_and_terminal_counts():
    """Verifies exactly 11 happy path states and 6 terminal reject states exist."""
    assert len(HAPPY_STATES) == 11
    assert len(TERMINAL_REJECT_STATES) == 6
    assert len(TERMINAL_STATES) == 7  # 6 rejects + 1 completed

    assert PipelineStatus.DISCOVERED in HAPPY_STATES
    assert PipelineStatus.COMPLETED in HAPPY_STATES
    assert PipelineStatus.REJECTED_LANGUAGE in TERMINAL_REJECT_STATES
    assert PipelineStatus.NO_GOOD_CLIP in TERMINAL_REJECT_STATES
    assert PipelineStatus.REJECTED_VISUAL in TERMINAL_REJECT_STATES
    assert PipelineStatus.RENDER_FAILED in TERMINAL_REJECT_STATES
    assert PipelineStatus.QC_FAILED in TERMINAL_REJECT_STATES
    assert PipelineStatus.UPLOAD_FAILED in TERMINAL_REJECT_STATES


def test_state_machine_sequential_happy_transitions():
    """Verifies that the entire 11-step happy progression transitions legally."""
    progression = [
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

    for i in range(len(progression) - 1):
        curr = progression[i]
        nxt = progression[i + 1]
        assert can_transition(curr, nxt) is True
        validated = validate_transition(curr, nxt)
        assert validated == nxt


def test_state_machine_guard_invariant_uploading_only_from_qc_passed():
    """Mandatory Guard Invariant: 'uploading' can ONLY transition from 'qc_passed'."""
    # From qc_passed: Allowed!
    assert can_transition(PipelineStatus.QC_PASSED, PipelineStatus.UPLOADING) is True
    assert validate_transition(PipelineStatus.QC_PASSED, PipelineStatus.UPLOADING) == PipelineStatus.UPLOADING

    # From any other state: Strictly Forbidden!
    illegal_origins = [
        PipelineStatus.DISCOVERED,
        PipelineStatus.ELIGIBLE,
        PipelineStatus.TRANSCRIBED,
        PipelineStatus.CANDIDATES_FOUND,
        PipelineStatus.CANDIDATE_SELECTED,
        PipelineStatus.VISUAL_VERIFIED,
        PipelineStatus.RENDERING,
        PipelineStatus.RENDERED,  # Crucial: Cannot skip QC to upload!
        PipelineStatus.QC_FAILED,
        PipelineStatus.COMPLETED,
    ]

    for origin in illegal_origins:
        assert can_transition(origin, PipelineStatus.UPLOADING) is False
        with pytest.raises(InvalidStateTransitionError):
            validate_transition(origin, PipelineStatus.UPLOADING)


def test_state_machine_terminal_states_cannot_transition():
    """Terminal states must never transition to any further state."""
    all_terminal = list(TERMINAL_STATES)
    for term in all_terminal:
        assert is_terminal(term) is True
        for target in PipelineStatus:
            assert can_transition(term, target) is False
            with pytest.raises(InvalidStateTransitionError):
                validate_transition(term, target)


def test_state_machine_rejection_branch_transitions():
    """Verifies that terminal rejection transitions are permitted from their proper stages."""
    # Language rejection
    assert can_transition(PipelineStatus.DISCOVERED, PipelineStatus.REJECTED_LANGUAGE) is True
    assert can_transition(PipelineStatus.ELIGIBLE, PipelineStatus.REJECTED_LANGUAGE) is True
    assert can_transition(PipelineStatus.TRANSCRIBED, PipelineStatus.REJECTED_LANGUAGE) is True

    # No good clip
    assert can_transition(PipelineStatus.CANDIDATES_FOUND, PipelineStatus.NO_GOOD_CLIP) is True
    assert can_transition(PipelineStatus.CANDIDATE_SELECTED, PipelineStatus.NO_GOOD_CLIP) is True

    # Visual rejection
    assert can_transition(PipelineStatus.CANDIDATE_SELECTED, PipelineStatus.REJECTED_VISUAL) is True
    assert can_transition(PipelineStatus.VISUAL_VERIFIED, PipelineStatus.REJECTED_VISUAL) is True

    # Render failed
    assert can_transition(PipelineStatus.RENDERING, PipelineStatus.RENDER_FAILED) is True

    # QC failed
    assert can_transition(PipelineStatus.RENDERED, PipelineStatus.QC_FAILED) is True

    # Upload failed
    assert can_transition(PipelineStatus.UPLOADING, PipelineStatus.UPLOAD_FAILED) is True


# ==============================================================================
# 2. Bab 18: Storage & Database (SQLite Repository) Tests
# ==============================================================================

def test_storage_repository_init_and_tables(temp_db: StorageRepository):
    """Verifies tables and indexes are created properly in SQLite."""
    with temp_db._get_connection() as conn:
        tables = [
            r["name"]
            for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall()
        ]
        assert "videos" in tables
        assert "candidates" in tables
        assert "renders" in tables
        assert "uploads" in tables

        indexes = [
            r["name"]
            for r in conn.execute("SELECT name FROM sqlite_master WHERE type='index';").fetchall()
        ]
        assert "idx_videos_status" in indexes
        assert "idx_candidates_video_id" in indexes
        assert "idx_renders_video_id" in indexes
        assert "idx_uploads_video_id" in indexes


def test_storage_video_crud_and_validated_transitions(temp_db: StorageRepository):
    """Verifies saving video and transitioning status with validation."""
    video = VideoRecord(
        video_id="test_vid_001",
        url="https://youtube.com/watch?v=test_vid_001",
        title="Podcast Inspiratif Indonesia",
        channel_title="Studio ID",
        duration_sec=600.0,
        status=PipelineStatus.DISCOVERED,
    )
    saved = temp_db.save_video(video)
    assert saved.video_id == "test_vid_001"
    assert saved.status == PipelineStatus.DISCOVERED

    # Legal transition: discovered -> eligible
    up1 = temp_db.update_video_status("test_vid_001", PipelineStatus.ELIGIBLE)
    assert up1.status == PipelineStatus.ELIGIBLE

    # Illegal transition: eligible -> uploading (skipping pipeline)
    with pytest.raises(InvalidStateTransitionError):
        temp_db.update_video_status("test_vid_001", PipelineStatus.UPLOADING)

    # Continue legal transitions
    temp_db.update_video_status("test_vid_001", PipelineStatus.TRANSCRIBED)
    temp_db.update_video_status("test_vid_001", PipelineStatus.CANDIDATES_FOUND)
    temp_db.update_video_status("test_vid_001", PipelineStatus.CANDIDATE_SELECTED)
    temp_db.update_video_status("test_vid_001", PipelineStatus.VISUAL_VERIFIED)
    temp_db.update_video_status("test_vid_001", PipelineStatus.RENDERING)
    temp_db.update_video_status("test_vid_001", PipelineStatus.RENDERED)
    temp_db.update_video_status("test_vid_001", PipelineStatus.QC_PASSED)
    temp_db.update_video_status("test_vid_001", PipelineStatus.UPLOADING)
    final = temp_db.update_video_status("test_vid_001", PipelineStatus.COMPLETED)
    assert final.status == PipelineStatus.COMPLETED


def test_storage_candidate_operations(temp_db: StorageRepository):
    """Verifies inserting candidates and marking selection."""
    video = VideoRecord(video_id="vid_cand_test", url="http://vid", status=PipelineStatus.DISCOVERED)
    temp_db.save_video(video)

    cand = CandidateRecord(
        candidate_id="cand_1",
        video_id="vid_cand_test",
        start_sec=10.0,
        end_sec=45.0,
        duration_sec=35.0,
        text="Ini cerita inspiratif tentang startup lokal.",
        overall_score=88.5,
        hook_score=90.0,
        payoff_score=87.0,
        self_contained_score=89.0,
    )
    cand_pk = temp_db.save_candidate(cand)
    assert cand_pk > 0

    fetched = temp_db.get_candidate(cand_pk)
    assert fetched is not None
    assert fetched.candidate_id == "cand_1"
    assert fetched.selected is False

    temp_db.mark_candidate_selected(cand_pk, visual_approved=True, visual_notes="Face centered")
    updated_cand = temp_db.get_candidate(cand_pk)
    assert updated_cand is not None
    assert updated_cand.selected is True
    assert updated_cand.visual_approved is True
    assert updated_cand.visual_notes == "Face centered"


def test_storage_render_and_upload_operations(temp_db: StorageRepository):
    """Verifies render and upload persistence."""
    video = VideoRecord(video_id="vid_media_test", url="http://vid", status=PipelineStatus.DISCOVERED)
    temp_db.save_video(video)

    # 1. Render record
    render = RenderRecord(
        video_id="vid_media_test",
        rendered_path="/tmp/test_render.mp4",
        duration_sec=42.0,
        status="rendering",
    )
    render_pk = temp_db.save_render(render)
    assert render_pk > 0

    temp_db.update_render_status(
        render_id=render_pk,
        status="rendered",
        qc_passed=True,
        qc_report_json=json.dumps({"passed": True}),
    )
    r_fetched = temp_db.get_render(render_pk)
    assert r_fetched is not None
    assert r_fetched.status == "rendered"
    assert r_fetched.qc_passed is True

    # 2. Upload record
    upload = UploadRecord(
        render_id=render_pk,
        video_id="vid_media_test",
        platform="youtube",
        status="uploading",
        title="Short Keren #Shorts",
        dry_run=True,
    )
    upload_pk = temp_db.save_upload(upload)
    assert upload_pk > 0

    temp_db.update_upload_status(
        upload_id=upload_pk,
        status="completed",
        platform_video_id="yt_12345",
        url="https://youtube.com/shorts/yt_12345",
    )
    uploads = temp_db.get_uploads_for_video("vid_media_test")
    assert len(uploads) == 1
    assert uploads[0].status == "completed"
    assert uploads[0].platform_video_id == "yt_12345"

    # 3. Complete relational history
    history = temp_db.get_video_history("vid_media_test")
    assert history is not None
    assert history["video"]["video_id"] == "vid_media_test"
    assert len(history["renders"]) == 1
    assert len(history["uploads"]) == 1


# ==============================================================================
# 3. Bab 17: Strict Upload Gate & YouTube Shorts Uploader Tests
# ==============================================================================

def test_strict_upload_gate_all_pass(valid_gate_kwargs: Dict[str, Any]):
    """Strict Upload Gate approves when all 8 gates pass."""
    assert all_gates_pass(**valid_gate_kwargs) is True

    gate_check = StrictUploadGate.evaluate(**valid_gate_kwargs, raise_on_failure=True)
    assert gate_check.passed is True
    assert len(gate_check.failed_gates()) == 0


def test_strict_upload_gate_rejects_each_failing_gate_individually(valid_gate_kwargs: Dict[str, Any]):
    """Strict Upload Gate MUST reject if any individual gate fails."""
    gate_names = list(valid_gate_kwargs.keys())

    for gate_to_fail in gate_names:
        broken_kwargs = valid_gate_kwargs.copy()
        broken_kwargs[gate_to_fail] = False

        assert all_gates_pass(**broken_kwargs) is False

        with pytest.raises(UploadGateRejectedError) as exc_info:
            StrictUploadGate.evaluate(**broken_kwargs, raise_on_failure=True)

        assert gate_to_fail in exc_info.value.failed_gates


def test_youtube_uploader_dry_run_success(dummy_video_file: Path, valid_gate_kwargs: Dict[str, Any]):
    """YouTubeShortsUploader succeeds in dry_run mode when gates pass."""
    uploader = YouTubeShortsUploader()
    gate_check = StrictUploadGate.evaluate(**valid_gate_kwargs)

    res = uploader.upload_short(
        video_path=dummy_video_file,
        title="Cerita Lucu Banget",
        description="Podcast viral seru",
        gate_check=gate_check,
        dry_run=True,
    )

    assert res["status"] == "success"
    assert res["dry_run"] is True
    assert res["platform"] == "youtube"
    assert "https://youtube.com/shorts/dry_run_" in res["url"]
    assert "#Shorts" in res["title"]


def test_youtube_uploader_rejects_without_gate_check(dummy_video_file: Path):
    """Uploader strictly forbids upload when gate_check is missing."""
    uploader = YouTubeShortsUploader()
    with pytest.raises(UploadGateRejectedError):
        uploader.upload_short(
            video_path=dummy_video_file,
            title="Video Tanpa Gate",
            description="Harus tolak",
            gate_check=None,
            dry_run=True,
        )


def test_youtube_uploader_rejects_failing_gate(dummy_video_file: Path, valid_gate_kwargs: Dict[str, Any]):
    """Uploader forbids upload when gate_check has any failed gate."""
    valid_gate_kwargs["perceptual_qc"] = False
    gate_check = StrictUploadGate.evaluate(**valid_gate_kwargs, raise_on_failure=False)

    uploader = YouTubeShortsUploader()
    with pytest.raises(UploadGateRejectedError):
        uploader.upload_short(
            video_path=dummy_video_file,
            title="Video QC Gagal",
            description="Harus tolak",
            gate_check=gate_check,
            dry_run=True,
        )


def test_youtube_uploader_title_formatting():
    """Title formatting guarantees #Shorts tag and <= 100 length."""
    uploader = YouTubeShortsUploader()

    # Appends #Shorts
    t1 = uploader.format_shorts_title("Kisah Sukses Programmer")
    assert t1.endswith("#Shorts")
    assert len(t1) <= 100

    # Retains existing #Shorts
    t2 = uploader.format_shorts_title("Kisah Sukses #Shorts Seru")
    assert t2 == "Kisah Sukses #Shorts Seru"

    # Truncates excessive title properly
    long_title = "A" * 150
    t3 = uploader.format_shorts_title(long_title)
    assert t3.endswith("#Shorts")
    assert len(t3) <= 100


# ==============================================================================
# 4. Bab 22: Full End-to-End Auto Clipper Orchestrator Tests
# ==============================================================================

def test_orchestrator_happy_path(tmp_path: Path, temp_db: StorageRepository, dummy_video_file: Path):
    """Full End-to-End Orchestrator Happy Path execution."""
    out_dir = tmp_path / "output"
    down_dir = tmp_path / "downloads"

    # Mock components
    mock_searcher = MagicMock()
    mock_searcher.get_video_metadata.return_value = VideoSourceMeta(
        video_id="happy_vid_01",
        url="https://youtube.com/watch?v=happy_vid_01",
        title="Podcast Lucu Indonesia",
        channel="StandUp Indo",
        duration_sec=400.0,
    )

    mock_source_filter = MagicMock()
    mock_source_filter.filter_video.return_value = EligibilityResult(
        is_eligible=True,
        reason="Video meets criteria",
    )

    mock_lang_gate = MagicMock()
    mock_lang_gate.evaluate_transcript.return_value = LanguageGateResult(
        eligible=True,
        primary_language="id",
        confidence=0.95,
        reason="Dominant Indonesian speech",
    )

    # Valid Indonesian transcript segments
    sample_transcript = [
        TranscriptSegment(start=0.0, end=5.0, duration=5.0, text="Gue waktu itu jalan ke Bandung bro."),
        TranscriptSegment(start=5.0, end=15.0, duration=10.0, text="Tiba-tiba ada hal aneh banget di jalan tol."),
        TranscriptSegment(start=15.0, end=35.0, duration=20.0, text="Semua orang ketawa ngakak denger ceritanya."),
        TranscriptSegment(start=35.0, end=42.0, duration=7.0, text="Dan akhirnya kita semua selamat."),
    ]
    mock_transcript_provider = MagicMock()
    mock_transcript_provider.get_phrase_transcript.return_value = sample_transcript

    mock_candidate_gen = MagicMock()
    mock_cand = CandidateWindow(
        candidate_id="cand_win_01",
        start_sec=5.0,
        end_sec=40.0,
        duration_sec=35.0,
        text="Tiba-tiba ada hal aneh banget di jalan tol.",
    )
    mock_candidate_gen.generate_candidates.return_value = [mock_cand]

    mock_scorer = MagicMock()
    mock_score = SemanticScore(
        candidate_id="cand_win_01",
        good_clip=True,
        score=85.0,
        hook_score=88.0,
        payoff_score=84.0,
        self_contained_score=86.0,
        reason="Strong hook and payoff",
        suggested_start=5.0,
        suggested_end=40.0,
    )
    mock_scorer.score_candidates.return_value = [mock_score]
    mock_scorer.select_best_clip.return_value = (mock_cand, mock_score, "APPROVED")

    mock_refiner = MagicMock()
    mock_refiner.refine.return_value = RefinementResult(
        is_valid=True,
        refined_start=5.0,
        refined_end=40.0,
        duration=35.0,
    )

    mock_vis_analyzer = MagicMock()
    mock_vis_analyzer.analyze_clip.return_value = VisualAnalysisReport(
        video_path=str(dummy_video_file),
        start_sec=5.0,
        end_sec=40.0,
        duration=35.0,
        scene_cuts=[15.0, 28.0],
        subject_presence_ratio=0.92,
        has_burned_subtitles=False,
    )

    mock_vis_director = MagicMock()
    mock_vis_director.evaluate_window.return_value = VisualDirectorVerdict(
        approved=True,
        confidence=0.95,
        shot_type="single_speaker",
        faces_visible=1,
        subject_visible_ratio=0.92,
        notes="Clean single speaker portrait candidate",
    )

    mock_framing = MagicMock()
    mock_framing.analyze_framing.return_value = FramingDecision(
        layout="PORTRAIT_9_16",
        crop_windows=[
            SceneCrop(scene_start=5.0, scene_end=15.0, crop_x=420, crop_y=0, crop_w=1080, crop_h=1920),
            SceneCrop(scene_start=15.0, scene_end=40.0, crop_x=420, crop_y=0, crop_w=1080, crop_h=1920),
        ],
    )

    mock_subtitle_cls = MagicMock()
    mock_subtitle_cls.check_burned_in_subtitles.return_value = (False, None)

    # Clean Renderer producing the dummy file
    mock_renderer = MagicMock()
    def fake_render(video_path, output_path, edit_plan, start_sec, duration, subtitle_ass_path=None):
        Path(output_path).write_bytes(b"\x00" * (120 * 1024))
        return RenderResult(
            output_path=str(output_path),
            duration=duration,
            success=True,
        )
    mock_renderer.render.side_effect = fake_render

    # Three-Tier QC Gate Passing
    mock_qc_gate = MagicMock()
    mock_qc_gate.evaluate.return_value = ThreeTierQCReport(
        passed=True,
        publishable=True,
        technical=TechnicalQCResult(
            passed=True,
            duration=35.0,
            width=1080,
            height=1920,
            video_codec="h264",
            audio_codec="aac",
            sample_rate=48000,
        ),
        visual=VisualQCResult(
            passed=True,
            sampled_frames_count=18,
            subject_present_ratio=0.92,
            subtitle_safe=True,
        ),
        perceptual=PerceptualQCResult(
            passed=True,
            publishable=True,
            score=88,
        ),
    )

    uploader = YouTubeShortsUploader()

    orchestrator = AutoClipperOrchestrator(
        repository=temp_db,
        searcher=mock_searcher,
        source_filter=mock_source_filter,
        language_gate=mock_lang_gate,
        transcript_provider=mock_transcript_provider,
        candidate_generator=mock_candidate_gen,
        semantic_scorer=mock_scorer,
        boundary_refiner=mock_refiner,
        visual_analyzer=mock_vis_analyzer,
        visual_director=mock_vis_director,
        framing=mock_framing,
        subtitle_classifier=mock_subtitle_cls,
        renderer=mock_renderer,
        qc_gate=mock_qc_gate,
        uploader=uploader,
        output_dir=out_dir,
        download_dir=down_dir,
    )

    result = orchestrator.process_video(
        video_id_or_url="happy_vid_01",
        dry_run=True,
        custom_video_path=str(dummy_video_file),
    )

    # Verifications
    assert result.is_success is True
    assert result.status == PipelineStatus.COMPLETED
    assert result.video_id == "happy_vid_01"
    assert result.duration_sec == 35.0
    assert result.gate_check is not None
    assert result.gate_check.passed is True
    assert result.upload_result is not None
    assert result.upload_result["status"] == "success"

    # Database verification
    db_vid = temp_db.get_video("happy_vid_01")
    assert db_vid is not None
    assert db_vid.status == PipelineStatus.COMPLETED

    candidates = temp_db.get_candidates_for_video("happy_vid_01")
    assert len(candidates) == 1
    assert candidates[0].selected is True

    renders = temp_db.get_renders_for_video("happy_vid_01")
    assert len(renders) == 1
    assert renders[0].qc_passed is True

    uploads = temp_db.get_uploads_for_video("happy_vid_01")
    assert len(uploads) == 1
    assert uploads[0].status == "dry_run"


def test_orchestrator_rejection_at_language_gate(temp_db: StorageRepository):
    """Orchestrator halts and records rejected_language when Indonesian gate fails."""
    mock_searcher = MagicMock()
    mock_searcher.get_video_metadata.return_value = VideoSourceMeta(
        video_id="english_vid_99",
        url="http://vid",
        title="English Tech Keynote",
        duration_sec=600.0,
    )

    mock_filter = MagicMock()
    mock_filter.filter_video.return_value = EligibilityResult(is_eligible=True, reason="OK")

    mock_transcript = MagicMock()
    mock_transcript.get_phrase_transcript.return_value = [
        TranscriptSegment(start=0.0, end=10.0, duration=10.0, text="Welcome everyone to our keynote speech today.")
    ]

    mock_lang = MagicMock()
    mock_lang.evaluate_transcript.return_value = LanguageGateResult(
        eligible=False,
        primary_language="en",
        confidence=0.98,
        reason="English dominance detected",
    )

    orchestrator = AutoClipperOrchestrator(
        repository=temp_db,
        searcher=mock_searcher,
        source_filter=mock_filter,
        transcript_provider=mock_transcript,
        language_gate=mock_lang,
    )

    result = orchestrator.process_video("english_vid_99", dry_run=True)

    assert result.is_success is False
    assert result.status == PipelineStatus.REJECTED_LANGUAGE
    assert result.rejection_reason is not None
    assert "English dominance" in result.rejection_reason

    db_vid = temp_db.get_video("english_vid_99")
    assert db_vid is not None
    assert db_vid.status == PipelineStatus.REJECTED_LANGUAGE


def test_orchestrator_rejection_at_semantic_scorer(temp_db: StorageRepository):
    """Orchestrator halts and records no_good_clip when semantic scorer finds no clip."""
    mock_searcher = MagicMock()
    mock_searcher.get_video_metadata.return_value = VideoSourceMeta(
        video_id="boring_vid_01",
        url="http://vid",
        title="Video Monoton Tanpa Hook",
        duration_sec=500.0,
    )
    mock_filter = MagicMock()
    mock_filter.filter_video.return_value = EligibilityResult(is_eligible=True, reason="OK")

    mock_transcript = MagicMock()
    mock_transcript.get_phrase_transcript.return_value = [
        TranscriptSegment(start=0.0, end=30.0, duration=30.0, text="Ya begitulah kira-kira hari ini tidak ada yang menarik.")
    ]

    mock_lang = MagicMock()
    mock_lang.evaluate_transcript.return_value = LanguageGateResult(eligible=True, primary_language="id", confidence=0.9, reason="Indonesian OK")

    mock_cand_gen = MagicMock()
    c = CandidateWindow(candidate_id="c1", start_sec=0.0, end_sec=30.0, duration_sec=30.0, text="...")
    mock_cand_gen.generate_candidates.return_value = [c]

    mock_scorer = MagicMock()
    mock_scorer.score_candidates.return_value = []
    mock_scorer.select_best_clip.return_value = (None, None, "NO GOOD CLIP FOUND")

    orchestrator = AutoClipperOrchestrator(
        repository=temp_db,
        searcher=mock_searcher,
        source_filter=mock_filter,
        transcript_provider=mock_transcript,
        language_gate=mock_lang,
        candidate_generator=mock_cand_gen,
        semantic_scorer=mock_scorer,
    )

    result = orchestrator.process_video("boring_vid_01", dry_run=True)

    assert result.is_success is False
    assert result.status == PipelineStatus.NO_GOOD_CLIP
    assert result.rejection_reason is not None
    assert "NO GOOD CLIP FOUND" in result.rejection_reason

    db_vid = temp_db.get_video("boring_vid_01")
    assert db_vid is not None
    assert db_vid.status == PipelineStatus.NO_GOOD_CLIP


def test_orchestrator_rejection_at_visual_director(temp_db: StorageRepository, dummy_video_file: Path):
    """Orchestrator halts and records rejected_visual when Visual Director rejects."""
    mock_searcher = MagicMock()
    mock_searcher.get_video_metadata.return_value = VideoSourceMeta(
        video_id="dark_vid_01",
        url="http://vid",
        title="Video Gelap Tanpa Wajah",
        duration_sec=400.0,
    )
    mock_filter = MagicMock()
    mock_filter.filter_video.return_value = EligibilityResult(is_eligible=True, reason="OK")

    mock_transcript = MagicMock()
    mock_transcript.get_phrase_transcript.return_value = [
        TranscriptSegment(start=10.0, end=45.0, duration=35.0, text="Halo teman-teman semua di Indonesia.")
    ]

    mock_lang = MagicMock()
    mock_lang.evaluate_transcript.return_value = LanguageGateResult(eligible=True, primary_language="id", confidence=0.9, reason="Indonesian OK")

    mock_cand_gen = MagicMock()
    c = CandidateWindow(candidate_id="c1", start_sec=10.0, end_sec=45.0, duration_sec=35.0, text="...")
    mock_cand_gen.generate_candidates.return_value = [c]

    mock_scorer = MagicMock()
    s = SemanticScore(
        candidate_id="c1",
        good_clip=True,
        score=80.0,
        hook_score=80.0,
        payoff_score=80.0,
        self_contained_score=80.0,
        reason="Good",
        suggested_start=10.0,
        suggested_end=45.0,
    )
    mock_scorer.score_candidates.return_value = [s]
    mock_scorer.select_best_clip.return_value = (c, s, "OK")

    mock_refiner = MagicMock()
    mock_refiner.refine.return_value = RefinementResult(is_valid=True, refined_start=10.0, refined_end=45.0, duration=35.0)

    mock_vis_analyzer = MagicMock()
    mock_vis_analyzer.analyze_clip.return_value = VisualAnalysisReport(
        video_path=str(dummy_video_file),
        start_sec=10.0,
        end_sec=45.0,
        duration=35.0,
        subject_presence_ratio=0.1,
    )

    mock_vis_director = MagicMock()
    mock_vis_director.evaluate_window.return_value = VisualDirectorVerdict(
        approved=False,
        rejection_reasons=["Subject missing in >30% duration", "Excessive black frames"],
    )

    orchestrator = AutoClipperOrchestrator(
        repository=temp_db,
        searcher=mock_searcher,
        source_filter=mock_filter,
        transcript_provider=mock_transcript,
        language_gate=mock_lang,
        candidate_generator=mock_cand_gen,
        semantic_scorer=mock_scorer,
        boundary_refiner=mock_refiner,
        visual_analyzer=mock_vis_analyzer,
        visual_director=mock_vis_director,
    )

    result = orchestrator.process_video(
        video_id_or_url="dark_vid_01",
        dry_run=True,
        custom_video_path=str(dummy_video_file),
    )

    assert result.is_success is False
    assert result.status == PipelineStatus.REJECTED_VISUAL
    assert result.rejection_reason is not None
    assert "Subject missing" in result.rejection_reason

    db_vid = temp_db.get_video("dark_vid_01")
    assert db_vid is not None
    assert db_vid.status == PipelineStatus.REJECTED_VISUAL


def test_orchestrator_rejection_at_qc_gate(temp_db: StorageRepository, dummy_video_file: Path, tmp_path: Path):
    """Orchestrator halts and records qc_failed when Quality Control gate fails."""
    mock_searcher = MagicMock()
    mock_searcher.get_video_metadata.return_value = VideoSourceMeta(
        video_id="corrupt_vid_01",
        url="http://vid",
        title="Video Gagal QC",
        duration_sec=400.0,
    )
    mock_filter = MagicMock()
    mock_filter.filter_video.return_value = EligibilityResult(is_eligible=True, reason="OK")

    mock_transcript = MagicMock()
    mock_transcript.get_phrase_transcript.return_value = [
        TranscriptSegment(start=10.0, end=45.0, duration=35.0, text="Halo teman-teman semua di Indonesia.")
    ]

    mock_lang = MagicMock()
    mock_lang.evaluate_transcript.return_value = LanguageGateResult(eligible=True, primary_language="id", confidence=0.9, reason="Indonesian OK")

    mock_cand_gen = MagicMock()
    c = CandidateWindow(candidate_id="c1", start_sec=10.0, end_sec=45.0, duration_sec=35.0, text="...")
    mock_cand_gen.generate_candidates.return_value = [c]

    mock_scorer = MagicMock()
    s = SemanticScore(
        candidate_id="c1",
        good_clip=True,
        score=80.0,
        hook_score=80.0,
        payoff_score=80.0,
        self_contained_score=80.0,
        reason="Good",
        suggested_start=10.0,
        suggested_end=45.0,
    )
    mock_scorer.score_candidates.return_value = [s]
    mock_scorer.select_best_clip.return_value = (c, s, "OK")

    mock_refiner = MagicMock()
    mock_refiner.refine.return_value = RefinementResult(is_valid=True, refined_start=10.0, refined_end=45.0, duration=35.0)

    mock_vis_analyzer = MagicMock()
    mock_vis_analyzer.analyze_clip.return_value = VisualAnalysisReport(
        video_path=str(dummy_video_file),
        start_sec=10.0,
        end_sec=45.0,
        duration=35.0,
        subject_presence_ratio=0.9,
    )

    mock_vis_director = MagicMock()
    mock_vis_director.evaluate_window.return_value = VisualDirectorVerdict(approved=True)

    mock_framing = MagicMock()
    mock_framing.analyze_framing.return_value = FramingDecision(layout="PORTRAIT_9_16")

    mock_renderer = MagicMock()
    def fake_render(video_path, output_path, edit_plan, start_sec, duration, subtitle_ass_path=None):
        Path(output_path).write_bytes(b"\x00" * (120 * 1024))
        return RenderResult(output_path=str(output_path), duration=duration, success=True)
    mock_renderer.render.side_effect = fake_render

    # QC FAILS
    mock_qc_gate = MagicMock()
    mock_qc_gate.evaluate.return_value = ThreeTierQCReport(
        passed=False,
        publishable=False,
        technical=TechnicalQCResult(
            passed=False,
            errors=["Corrupt video stream detected"],
        ),
        visual=VisualQCResult(passed=True),
        perceptual=PerceptualQCResult(passed=True, publishable=True, score=85),
        errors=["Corrupt video stream detected"],
    )

    orchestrator = AutoClipperOrchestrator(
        repository=temp_db,
        searcher=mock_searcher,
        source_filter=mock_filter,
        transcript_provider=mock_transcript,
        language_gate=mock_lang,
        candidate_generator=mock_cand_gen,
        semantic_scorer=mock_scorer,
        boundary_refiner=mock_refiner,
        visual_analyzer=mock_vis_analyzer,
        visual_director=mock_vis_director,
        framing=mock_framing,
        renderer=mock_renderer,
        qc_gate=mock_qc_gate,
        output_dir=tmp_path / "output",
        download_dir=tmp_path / "downloads",
    )

    result = orchestrator.process_video(
        video_id_or_url="corrupt_vid_01",
        dry_run=True,
        custom_video_path=str(dummy_video_file),
    )

    assert result.is_success is False
    assert result.status == PipelineStatus.QC_FAILED
    assert result.error_message is not None
    assert "Corrupt video stream detected" in result.error_message

    db_vid = temp_db.get_video("corrupt_vid_01")
    assert db_vid is not None
    assert db_vid.status == PipelineStatus.QC_FAILED


def test_orchestrator_discovery_cycle(temp_db: StorageRepository):
    """Verifies run_discovery_cycle discovers multiple videos and processes them."""
    mock_searcher = MagicMock()
    mock_searcher.search_eligible_videos.return_value = [
        VideoSourceMeta(video_id="v_cycle_1", url="http://v1", duration_sec=300.0),
        VideoSourceMeta(video_id="v_cycle_2", url="http://v2", duration_sec=300.0),
    ]

    orchestrator = AutoClipperOrchestrator(
        repository=temp_db,
        searcher=mock_searcher,
    )

    # Mock process_video
    with patch.object(orchestrator, "process_video") as mock_pv:
        mock_pv.side_effect = [
            PipelineResult(video_id="v_cycle_1", status=PipelineStatus.COMPLETED, is_success=True),
            PipelineResult(video_id="v_cycle_2", status=PipelineStatus.REJECTED_LANGUAGE, is_success=False),
        ]

        results = orchestrator.run_discovery_cycle(query="podcast viral", max_videos=2, dry_run=True)

        assert len(results) == 2
        assert results[0].video_id == "v_cycle_1"
        assert results[0].status == PipelineStatus.COMPLETED
        assert results[1].video_id == "v_cycle_2"
        assert results[1].status == PipelineStatus.REJECTED_LANGUAGE


# ==============================================================================
# 5. Additional Edge Cases & Negative Path Tests
# ==============================================================================

def test_orchestrator_skip_upload_transitions_to_completed(
    tmp_path: Path, temp_db: StorageRepository, dummy_video_file: Path
):
    """When skip_upload=True, orchestrator transitions directly from qc_passed to completed."""
    mock_searcher = MagicMock()
    mock_searcher.get_video_metadata.return_value = VideoSourceMeta(
        video_id="skip_up_vid", url="http://vid", duration_sec=300.0
    )
    mock_filter = MagicMock()
    mock_filter.filter_video.return_value = EligibilityResult(is_eligible=True, reason="OK")

    mock_transcript = MagicMock()
    mock_transcript.get_phrase_transcript.return_value = [
        TranscriptSegment(start=5.0, end=40.0, duration=35.0, text="Ini konten berbahasa Indonesia.")
    ]
    mock_lang = MagicMock()
    mock_lang.evaluate_transcript.return_value = LanguageGateResult(eligible=True, primary_language="id", confidence=0.9, reason="OK")

    mock_cand_gen = MagicMock()
    c = CandidateWindow(candidate_id="c1", start_sec=5.0, end_sec=40.0, duration_sec=35.0, text="...")
    mock_cand_gen.generate_candidates.return_value = [c]

    mock_scorer = MagicMock()
    s = SemanticScore(
        candidate_id="c1", good_clip=True, score=80.0, hook_score=80.0,
        payoff_score=80.0, self_contained_score=80.0, reason="Good", suggested_start=5.0, suggested_end=40.0
    )
    mock_scorer.score_candidates.return_value = [s]
    mock_scorer.select_best_clip.return_value = (c, s, "OK")

    mock_refiner = MagicMock()
    mock_refiner.refine.return_value = RefinementResult(is_valid=True, refined_start=5.0, refined_end=40.0, duration=35.0)

    mock_vis_analyzer = MagicMock()
    mock_vis_analyzer.analyze_clip.return_value = VisualAnalysisReport(
        video_path=str(dummy_video_file), start_sec=5.0, end_sec=40.0, duration=35.0, subject_presence_ratio=0.9
    )
    mock_vis_director = MagicMock()
    mock_vis_director.evaluate_window.return_value = VisualDirectorVerdict(approved=True)

    mock_framing = MagicMock()
    mock_framing.analyze_framing.return_value = FramingDecision(layout="PORTRAIT_9_16")

    mock_renderer = MagicMock()
    def fake_render(video_path, output_path, edit_plan, start_sec, duration, subtitle_ass_path=None):
        Path(output_path).write_bytes(b"\x00" * (120 * 1024))
        return RenderResult(output_path=str(output_path), duration=duration, success=True)
    mock_renderer.render.side_effect = fake_render

    mock_qc = MagicMock()
    mock_qc.evaluate.return_value = ThreeTierQCReport(
        passed=True, publishable=True,
        technical=TechnicalQCResult(passed=True, duration=35.0, width=1080, height=1920, video_codec="h264", audio_codec="aac", sample_rate=48000),
        visual=VisualQCResult(passed=True, subject_present_ratio=0.9, subtitle_safe=True),
        perceptual=PerceptualQCResult(passed=True, publishable=True, score=90),
    )

    uploader_mock = MagicMock()

    orchestrator = AutoClipperOrchestrator(
        repository=temp_db,
        searcher=mock_searcher,
        source_filter=mock_filter,
        transcript_provider=mock_transcript,
        language_gate=mock_lang,
        candidate_generator=mock_cand_gen,
        semantic_scorer=mock_scorer,
        boundary_refiner=mock_refiner,
        visual_analyzer=mock_vis_analyzer,
        visual_director=mock_vis_director,
        framing=mock_framing,
        renderer=mock_renderer,
        qc_gate=mock_qc,
        uploader=uploader_mock,
        output_dir=tmp_path / "output",
        download_dir=tmp_path / "downloads",
    )

    res = orchestrator.process_video(
        video_id_or_url="skip_up_vid",
        dry_run=True,
        custom_video_path=str(dummy_video_file),
        skip_upload=True,
    )

    assert res.is_success is True
    assert res.status == PipelineStatus.COMPLETED
    uploader_mock.upload_short.assert_not_called()

    db_vid = temp_db.get_video("skip_up_vid")
    assert db_vid is not None
    assert db_vid.status == PipelineStatus.COMPLETED


def test_orchestrator_upload_failure_transitions_to_upload_failed(
    tmp_path: Path, temp_db: StorageRepository, dummy_video_file: Path
):
    """When uploader fails, orchestrator records status as upload_failed."""
    mock_searcher = MagicMock()
    mock_searcher.get_video_metadata.return_value = VideoSourceMeta(
        video_id="fail_up_vid", url="http://vid", duration_sec=300.0
    )
    mock_filter = MagicMock()
    mock_filter.filter_video.return_value = EligibilityResult(is_eligible=True, reason="OK")

    mock_transcript = MagicMock()
    mock_transcript.get_phrase_transcript.return_value = [
        TranscriptSegment(start=5.0, end=40.0, duration=35.0, text="Ini konten berbahasa Indonesia.")
    ]
    mock_lang = MagicMock()
    mock_lang.evaluate_transcript.return_value = LanguageGateResult(eligible=True, primary_language="id", confidence=0.9, reason="OK")

    mock_cand_gen = MagicMock()
    c = CandidateWindow(candidate_id="c1", start_sec=5.0, end_sec=40.0, duration_sec=35.0, text="...")
    mock_cand_gen.generate_candidates.return_value = [c]

    mock_scorer = MagicMock()
    s = SemanticScore(
        candidate_id="c1", good_clip=True, score=80.0, hook_score=80.0,
        payoff_score=80.0, self_contained_score=80.0, reason="Good", suggested_start=5.0, suggested_end=40.0
    )
    mock_scorer.score_candidates.return_value = [s]
    mock_scorer.select_best_clip.return_value = (c, s, "OK")

    mock_refiner = MagicMock()
    mock_refiner.refine.return_value = RefinementResult(is_valid=True, refined_start=5.0, refined_end=40.0, duration=35.0)

    mock_vis_analyzer = MagicMock()
    mock_vis_analyzer.analyze_clip.return_value = VisualAnalysisReport(
        video_path=str(dummy_video_file), start_sec=5.0, end_sec=40.0, duration=35.0, subject_presence_ratio=0.9
    )
    mock_vis_director = MagicMock()
    mock_vis_director.evaluate_window.return_value = VisualDirectorVerdict(approved=True)

    mock_framing = MagicMock()
    mock_framing.analyze_framing.return_value = FramingDecision(layout="PORTRAIT_9_16")

    mock_renderer = MagicMock()
    def fake_render(video_path, output_path, edit_plan, start_sec, duration, subtitle_ass_path=None):
        Path(output_path).write_bytes(b"\x00" * (120 * 1024))
        return RenderResult(output_path=str(output_path), duration=duration, success=True)
    mock_renderer.render.side_effect = fake_render

    mock_qc = MagicMock()
    mock_qc.evaluate.return_value = ThreeTierQCReport(
        passed=True, publishable=True,
        technical=TechnicalQCResult(passed=True, duration=35.0, width=1080, height=1920, video_codec="h264", audio_codec="aac", sample_rate=48000),
        visual=VisualQCResult(passed=True, subject_present_ratio=0.9, subtitle_safe=True),
        perceptual=PerceptualQCResult(passed=True, publishable=True, score=90),
    )

    mock_uploader = MagicMock()
    mock_uploader.upload_short.return_value = {
        "status": "failed",
        "error": "OAuth token quota exceeded",
    }

    orchestrator = AutoClipperOrchestrator(
        repository=temp_db,
        searcher=mock_searcher,
        source_filter=mock_filter,
        transcript_provider=mock_transcript,
        language_gate=mock_lang,
        candidate_generator=mock_cand_gen,
        semantic_scorer=mock_scorer,
        boundary_refiner=mock_refiner,
        visual_analyzer=mock_vis_analyzer,
        visual_director=mock_vis_director,
        framing=mock_framing,
        renderer=mock_renderer,
        qc_gate=mock_qc,
        uploader=mock_uploader,
        output_dir=tmp_path / "output",
        download_dir=tmp_path / "downloads",
    )

    res = orchestrator.process_video(
        video_id_or_url="fail_up_vid",
        dry_run=False,
        custom_video_path=str(dummy_video_file),
    )

    assert res.is_success is False
    assert res.status == PipelineStatus.UPLOAD_FAILED
    assert "quota exceeded" in (res.error_message or "")

    db_vid = temp_db.get_video("fail_up_vid")
    assert db_vid is not None
    assert db_vid.status == PipelineStatus.UPLOAD_FAILED


def test_orchestrator_render_failure_transitions_to_render_failed(
    tmp_path: Path, temp_db: StorageRepository, dummy_video_file: Path
):
    """When renderer fails, orchestrator records status as render_failed."""
    mock_searcher = MagicMock()
    mock_searcher.get_video_metadata.return_value = VideoSourceMeta(
        video_id="fail_render_vid", url="http://vid", duration_sec=300.0
    )
    mock_filter = MagicMock()
    mock_filter.filter_video.return_value = EligibilityResult(is_eligible=True, reason="OK")

    mock_transcript = MagicMock()
    mock_transcript.get_phrase_transcript.return_value = [
        TranscriptSegment(start=5.0, end=40.0, duration=35.0, text="Ini konten berbahasa Indonesia.")
    ]
    mock_lang = MagicMock()
    mock_lang.evaluate_transcript.return_value = LanguageGateResult(eligible=True, primary_language="id", confidence=0.9, reason="OK")

    mock_cand_gen = MagicMock()
    c = CandidateWindow(candidate_id="c1", start_sec=5.0, end_sec=40.0, duration_sec=35.0, text="...")
    mock_cand_gen.generate_candidates.return_value = [c]

    mock_scorer = MagicMock()
    s = SemanticScore(
        candidate_id="c1", good_clip=True, score=80.0, hook_score=80.0,
        payoff_score=80.0, self_contained_score=80.0, reason="Good", suggested_start=5.0, suggested_end=40.0
    )
    mock_scorer.score_candidates.return_value = [s]
    mock_scorer.select_best_clip.return_value = (c, s, "OK")

    mock_refiner = MagicMock()
    mock_refiner.refine.return_value = RefinementResult(is_valid=True, refined_start=5.0, refined_end=40.0, duration=35.0)

    mock_vis_analyzer = MagicMock()
    mock_vis_analyzer.analyze_clip.return_value = VisualAnalysisReport(
        video_path=str(dummy_video_file), start_sec=5.0, end_sec=40.0, duration=35.0, subject_presence_ratio=0.9
    )
    mock_vis_director = MagicMock()
    mock_vis_director.evaluate_window.return_value = VisualDirectorVerdict(approved=True)

    mock_framing = MagicMock()
    mock_framing.analyze_framing.return_value = FramingDecision(layout="PORTRAIT_9_16")

    # Renderer FAILS
    mock_renderer = MagicMock()
    mock_renderer.render.return_value = RenderResult(
        output_path=str(tmp_path / "output" / "bad.mp4"),
        success=False,
        error_message="FFmpeg filter complex broken syntax",
    )

    orchestrator = AutoClipperOrchestrator(
        repository=temp_db,
        searcher=mock_searcher,
        source_filter=mock_filter,
        transcript_provider=mock_transcript,
        language_gate=mock_lang,
        candidate_generator=mock_cand_gen,
        semantic_scorer=mock_scorer,
        boundary_refiner=mock_refiner,
        visual_analyzer=mock_vis_analyzer,
        visual_director=mock_vis_director,
        framing=mock_framing,
        renderer=mock_renderer,
        output_dir=tmp_path / "output",
        download_dir=tmp_path / "downloads",
    )

    res = orchestrator.process_video(
        video_id_or_url="fail_render_vid",
        dry_run=True,
        custom_video_path=str(dummy_video_file),
    )

    assert res.is_success is False
    assert res.status == PipelineStatus.RENDER_FAILED
    assert "broken syntax" in (res.error_message or "")

    db_vid = temp_db.get_video("fail_render_vid")
    assert db_vid is not None
    assert db_vid.status == PipelineStatus.RENDER_FAILED


def test_youtube_uploader_small_and_missing_file_rejection(tmp_path: Path, valid_gate_kwargs: Dict[str, Any]):
    """YouTubeShortsUploader rejects missing files and files < 100KB."""
    uploader = YouTubeShortsUploader()
    gate_check = StrictUploadGate.evaluate(**valid_gate_kwargs)

    # 1. Non-existent file
    with pytest.raises(FileNotFoundError):
        uploader.upload_short(
            video_path=tmp_path / "non_existent.mp4",
            title="Judul",
            description="Deskripsi",
            gate_check=gate_check,
            dry_run=True,
        )

    # 2. Too small (< 100KB)
    small_file = tmp_path / "small.mp4"
    small_file.write_bytes(b"\x00" * 500)
    with pytest.raises(ValueError) as exc_info:
        uploader.upload_short(
            video_path=small_file,
            title="Judul",
            description="Deskripsi",
            gate_check=gate_check,
            dry_run=True,
        )
    assert "too small" in str(exc_info.value)


def test_storage_repository_list_and_history_edge_cases(temp_db: StorageRepository):
    """Verifies list_videos filtering and nonexistent history query."""
    v1 = VideoRecord(video_id="v_filter_1", url="http://1", status=PipelineStatus.DISCOVERED)
    v2 = VideoRecord(video_id="v_filter_2", url="http://2", status=PipelineStatus.COMPLETED)
    temp_db.save_video(v1)
    temp_db.save_video(v2)

    # Filter by status
    completed_videos = temp_db.list_videos(status=PipelineStatus.COMPLETED)
    assert len(completed_videos) == 1
    assert completed_videos[0].video_id == "v_filter_2"

    discovered_videos = temp_db.list_videos(status=PipelineStatus.DISCOVERED)
    assert len(discovered_videos) == 1
    assert discovered_videos[0].video_id == "v_filter_1"

    # Non-existent history
    assert temp_db.get_video_history("missing_vid_xyz") is None


def test_state_machine_parse_unknown_state():
    """parse_state raises ValueError on unknown state strings."""
    with pytest.raises(ValueError) as exc_info:
        parse_state("alien_state")
    assert "Unknown pipeline state" in str(exc_info.value)

