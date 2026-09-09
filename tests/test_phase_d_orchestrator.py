"""Unit Test Suite for Phase D: Uploader and Pipeline Orchestrator.

Blueprint Coverage:
- Bab 17: Strict Upload Gate & YouTube Shorts Uploader
  * all_gates_pass across all 8 production gates
  * Rejection enforcement (raise and return status)
  * YouTube Shorts uploader (dry_run, format_shorts_title, file validation)
  * DB state transitions: uploading -> completed | upload_failed
- Bab 3 & 22: Full End-to-End Auto Clipper Orchestrator
  * Stage 1 (Discovery & Source Filter) -> Stage 2 (Indonesian Gate) ->
    Stage 3 (Transcript) -> Stage 4 (Candidate Generator) ->
    Stage 5 (Semantic Scorer) -> Stage 6 (Visual Viability) ->
    Stage 7 (Boundary Refiner) -> Stage 8 (Edit Plan & Renderer) ->
    Stage 9 (Three-Tier QC) -> Stage 10 (Upload Gate)
  * process_video(video_meta_or_url)
  * run_discovery_cycle(search_queries, max_videos)
  * Core philosophy: "Reject Is a Valid Outcome"
"""

from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, patch
import pytest

from pipeline.state_machine import PipelineStatus
from storage.repository import StorageRepository, VideoRecord
from upload.uploader import (
    all_gates_pass,
    check_upload_gate,
    StrictUploadGate,
    UploadGateCheck,
    UploadGateRejectedError,
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
from editing.edit_plan import SceneCrop
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
    """Isolated SQLite repository for each test."""
    db_file = tmp_path / "test_orchestrator.db"
    return StorageRepository(db_path=db_file)


@pytest.fixture
def dummy_video_file(tmp_path: Path) -> Path:
    """Creates a dummy video file > 100KB."""
    f = tmp_path / "sample_video.mp4"
    f.write_bytes(b"\x00" * (128 * 1024))
    return f


@pytest.fixture
def all_gates_true() -> Dict[str, bool]:
    """Dictionary representing all 8 gates satisfied."""
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


# ==============================================================================
# 1. Strict Upload Gate & all_gates_pass Tests (Bab 17)
# ==============================================================================

def test_all_gates_pass_complete_matrix(all_gates_true):
    """Verifies all_gates_pass returns True ONLY when all 8 gates pass."""
    assert all_gates_pass(**all_gates_true) is True

    # Individually toggle each gate to False
    for gate_name in all_gates_true.keys():
        broken = dict(all_gates_true)
        broken[gate_name] = False
        assert all_gates_pass(**broken) is False, f"Gate '{gate_name}' failed to enforce rejection"


def test_strict_upload_gate_evaluate_success(all_gates_true):
    """StrictUploadGate.evaluate returns passed UploadGateCheck when all gates pass."""
    check = StrictUploadGate.evaluate(**all_gates_true, details={"video_id": "test_01"})
    assert isinstance(check, UploadGateCheck)
    assert check.passed is True
    assert check.failed_gates() == []
    assert check.details["video_id"] == "test_01"


def test_strict_upload_gate_evaluate_raises_on_failure(all_gates_true):
    """StrictUploadGate.evaluate raises UploadGateRejectedError when any gate fails with raise_on_failure=True."""
    broken = dict(all_gates_true)
    broken["technical_qc"] = False
    broken["perceptual_qc"] = False

    with pytest.raises(UploadGateRejectedError) as exc_info:
        StrictUploadGate.evaluate(**broken, raise_on_failure=True)

    err = exc_info.value
    assert "technical_qc" in err.failed_gates
    assert "perceptual_qc" in err.failed_gates
    assert "Strict Upload Gate REJECTED" in str(err)


def test_strict_upload_gate_evaluate_no_raise_returns_check(all_gates_true):
    """StrictUploadGate.evaluate returns check object with passed=False when raise_on_failure=False."""
    broken = dict(all_gates_true)
    broken["visual_viability_gate"] = False

    check = StrictUploadGate.evaluate(**broken, raise_on_failure=False)
    assert isinstance(check, UploadGateCheck)
    assert check.passed is False
    assert "visual_viability_gate" in check.failed_gates()


def test_check_upload_gate_helper(all_gates_true):
    """check_upload_gate convenience function works for pass and non-raising fail."""
    passed_check = check_upload_gate(**all_gates_true)
    assert passed_check.passed is True

    broken = dict(all_gates_true)
    broken["language_gate"] = False
    failed_check = check_upload_gate(**broken, raise_on_failure=False)
    assert failed_check.passed is False
    assert "language_gate" in failed_check.failed_gates()


# ==============================================================================
# 2. YouTube Shorts Uploader Tests (Bab 17)
# ==============================================================================

def test_uploader_title_formatting():
    """Uploader formats titles ensuring #Shorts is present and capped at 100 characters."""
    uploader = YouTubeShortsUploader()

    assert uploader.format_shorts_title("Cerita Lucu") == "Cerita Lucu #Shorts"
    assert uploader.format_shorts_title("Cerita Lucu #Shorts") == "Cerita Lucu #Shorts"
    assert uploader.format_shorts_title("Cerita Lucu #shorts") == "Cerita Lucu #shorts"
    assert uploader.format_shorts_title("") == "Video Viral Indonesia #Shorts"

    long_title = "A" * 120
    formatted = uploader.format_shorts_title(long_title)
    assert len(formatted) <= 100
    assert formatted.endswith("#Shorts")


def test_uploader_dry_run_success(dummy_video_file, all_gates_true):
    """Dry run upload succeeds and produces mock URL and metadata."""
    uploader = YouTubeShortsUploader()
    gate_check = StrictUploadGate.evaluate(**all_gates_true)

    result = uploader.upload_short(
        video_path=dummy_video_file,
        title="Podcast Lucu",
        description="Deskripsi podcast",
        gate_check=gate_check,
        dry_run=True,
    )

    assert result["status"] == "success"
    assert result["platform"] == "youtube"
    assert result["dry_run"] is True
    assert "https://youtube.com/shorts/dry_run_" in result["url"]
    assert "#Shorts" in result["title"]


def test_uploader_rejects_missing_or_failed_gate_check(dummy_video_file, all_gates_true):
    """Uploader rejects immediately when gate_check is None or has failing gates."""
    uploader = YouTubeShortsUploader()

    # None gate check
    with pytest.raises(UploadGateRejectedError) as exc_info:
        uploader.upload_short(
            video_path=dummy_video_file,
            title="Test",
            description="Test",
            gate_check=None,
            dry_run=True,
        )
    assert "ALL_GATES_MISSING" in exc_info.value.failed_gates

    # Failed gate check
    broken = dict(all_gates_true)
    broken["semantic_clip_gate"] = False
    failed_check = StrictUploadGate.evaluate(**broken, raise_on_failure=False)

    with pytest.raises(UploadGateRejectedError) as exc_info:
        uploader.upload_short(
            video_path=dummy_video_file,
            title="Test",
            description="Test",
            gate_check=failed_check,
            dry_run=True,
        )
    assert "semantic_clip_gate" in exc_info.value.failed_gates


def test_uploader_non_raising_rejection(dummy_video_file, all_gates_true):
    """Uploader returns rejection dict when raise_on_failure=False."""
    uploader = YouTubeShortsUploader()
    broken = dict(all_gates_true)
    broken["boundary_gate"] = False
    failed_check = StrictUploadGate.evaluate(**broken, raise_on_failure=False)

    res = uploader.upload_short(
        video_path=dummy_video_file,
        title="Test",
        description="Test",
        gate_check=failed_check,
        dry_run=True,
        raise_on_failure=False,
    )
    assert res["status"] == "rejected"
    assert "boundary_gate" in res["failed_gates"]


def test_uploader_rejects_nonexistent_and_small_files(tmp_path, all_gates_true):
    """Uploader rejects files that do not exist or are < 100KB."""
    uploader = YouTubeShortsUploader()
    gate_check = StrictUploadGate.evaluate(**all_gates_true)

    # Missing file
    with pytest.raises(FileNotFoundError):
        uploader.upload_short(
            video_path=tmp_path / "ghost.mp4",
            title="Ghost",
            description="Desc",
            gate_check=gate_check,
            dry_run=True,
        )

    # File too small (<100KB)
    tiny_file = tmp_path / "tiny.mp4"
    tiny_file.write_bytes(b"\x00" * 1024)
    with pytest.raises(ValueError) as exc_info:
        uploader.upload_short(
            video_path=tiny_file,
            title="Tiny",
            description="Desc",
            gate_check=gate_check,
            dry_run=True,
        )
    assert "too small" in str(exc_info.value)


def test_uploader_db_state_transitions(temp_db, dummy_video_file, all_gates_true):
    """Uploader transitions video in DB from uploading to completed or upload_failed."""
    uploader = YouTubeShortsUploader(repository=temp_db)
    gate_check = StrictUploadGate.evaluate(**all_gates_true)

    # Seed video in DB
    vid_id = "v_upload_test"
    temp_db.save_video(VideoRecord(video_id=vid_id, url="http://v", title="Test", status=PipelineStatus.QC_PASSED))

    # Successful upload transitions to completed
    uploader.upload_short(
        video_path=dummy_video_file,
        title="Test Short",
        description="Desc",
        gate_check=gate_check,
        dry_run=True,
        video_id=vid_id,
    )
    v_record = temp_db.get_video(vid_id)
    assert v_record is not None
    assert v_record.status == PipelineStatus.COMPLETED

    # Failed upload transitions to upload_failed
    vid_fail = "v_upload_fail"
    temp_db.save_video(VideoRecord(video_id=vid_fail, url="http://v", title="Fail", status=PipelineStatus.QC_PASSED))
    broken = dict(all_gates_true)
    broken["render_success"] = False
    bad_check = StrictUploadGate.evaluate(**broken, raise_on_failure=False)

    uploader.upload_short(
        video_path=dummy_video_file,
        title="Test Fail",
        description="Desc",
        gate_check=bad_check,
        dry_run=True,
        video_id=vid_fail,
        raise_on_failure=False,
    )
    v_fail_rec = temp_db.get_video(vid_fail)
    assert v_fail_rec is not None
    assert v_fail_rec.status == PipelineStatus.UPLOAD_FAILED


# ==============================================================================
# 3. Pipeline Orchestrator Tests (Bab 3 & 22)
# ==============================================================================

def test_orchestrator_resolve_video_id_variants():
    """_resolve_video_id correctly parses video IDs from various inputs."""
    orch = AutoClipperOrchestrator()

    # 1. VideoSourceMeta
    meta = VideoSourceMeta(video_id="id_meta_123", url="http://vid", title="T", duration_sec=100.0)
    assert orch._resolve_video_id(meta) == "id_meta_123"

    # 2. Dictionary
    assert orch._resolve_video_id({"video_id": "id_dict_456"}) == "id_dict_456"

    # 3. Standard YouTube URL
    assert orch._resolve_video_id("https://www.youtube.com/watch?v=dQw4w9WgXcQ") == "dQw4w9WgXcQ"

    # 4. Shortened youtu.be URL
    assert orch._resolve_video_id("https://youtu.be/dQw4w9WgXcQ?t=10") == "dQw4w9WgXcQ"

    # 5. Raw ID
    assert orch._resolve_video_id("dQw4w9WgXcQ") == "dQw4w9WgXcQ"


def _build_mock_orchestrator_harness(tmp_path: Path, temp_db: StorageRepository, dummy_video_file: Path):
    """Builds a mock-injected AutoClipperOrchestrator ready for scenario testing."""
    mock_searcher = MagicMock()
    mock_source_filter = MagicMock()
    mock_lang_gate = MagicMock()
    mock_transcript_provider = MagicMock()
    mock_candidate_gen = MagicMock()
    mock_scorer = MagicMock()
    mock_refiner = MagicMock()
    mock_vis_analyzer = MagicMock()
    mock_vis_director = MagicMock()
    mock_framing = MagicMock()
    mock_subtitle_cls = MagicMock()
    mock_renderer = MagicMock()
    mock_qc_gate = MagicMock()
    mock_uploader = MagicMock()
    mock_search_planner = MagicMock()
    mock_search_planner.plan_searches.return_value = MagicMock(queries=["podcast viral indonesia"])
    mock_vis_preflight = MagicMock()
    mock_vis_preflight.preflight_clip.return_value = MagicMock(
        usable=True,
        existing_visible_subtitles=False,
        recommended_layout="SAFE_WIDE",
        blocking_issues=[],
    )
    mock_gemini_video_qc = MagicMock()
    mock_gemini_video_qc.evaluate_video.return_value = MagicMock(
        passed=True,
        score=85,
        blocking_reasons=[],
    )

    out_dir = tmp_path / "output"
    down_dir = tmp_path / "downloads"
    out_dir.mkdir(parents=True, exist_ok=True)
    down_dir.mkdir(parents=True, exist_ok=True)

    orch = AutoClipperOrchestrator(
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
        uploader=mock_uploader,
        search_planner=mock_search_planner,
        visual_preflight=mock_vis_preflight,
        gemini_video_qc=mock_gemini_video_qc,
        output_dir=out_dir,
        download_dir=down_dir,
    )

    return (
        orch,
        {
            "searcher": mock_searcher,
            "source_filter": mock_source_filter,
            "language_gate": mock_lang_gate,
            "transcript_provider": mock_transcript_provider,
            "candidate_generator": mock_candidate_gen,
            "semantic_scorer": mock_scorer,
            "boundary_refiner": mock_refiner,
            "visual_analyzer": mock_vis_analyzer,
            "visual_director": mock_vis_director,
            "framing": mock_framing,
            "subtitle_classifier": mock_subtitle_cls,
            "renderer": mock_renderer,
            "qc_gate": mock_qc_gate,
            "uploader": mock_uploader,
            "visual_preflight": mock_vis_preflight,
            "gemini_video_qc": mock_gemini_video_qc,
        },
    )


def test_orchestrator_happy_path_full_pipeline(tmp_path, temp_db, dummy_video_file):
    """End-to-End Orchestrator Happy Path passing all 10 stages."""
    orch, mocks = _build_mock_orchestrator_harness(tmp_path, temp_db, dummy_video_file)

    vid_meta = VideoSourceMeta(
        video_id="happy_vid_99",
        url="https://youtube.com/watch?v=happy_vid_99",
        title="Podcast Keren Indonesia",
        channel="IndoCast",
        duration_sec=360.0,
    )
    mocks["searcher"].get_video_metadata.return_value = vid_meta
    mocks["source_filter"].filter_video.return_value = EligibilityResult(is_eligible=True, reason="Eligible")

    transcript = [
        TranscriptSegment(start=0.0, end=10.0, duration=10.0, text="Ini adalah cerita luar biasa."),
        TranscriptSegment(start=10.0, end=45.0, duration=35.0, text="Waktu di Jakarta kita nemu tempat ini mantap."),
    ]
    mocks["transcript_provider"].get_phrase_transcript.return_value = transcript
    mocks["language_gate"].evaluate_transcript.return_value = LanguageGateResult(
        eligible=True, primary_language="id", confidence=0.98, reason="Indonesian dominant"
    )

    candidate = CandidateWindow(
        candidate_id="cand_99",
        start_sec=10.0,
        end_sec=45.0,
        duration_sec=35.0,
        text="Waktu di Jakarta kita nemu tempat ini mantap.",
    )
    mocks["candidate_generator"].generate_candidates.return_value = [candidate]

    score = SemanticScore(
        candidate_id="cand_99",
        good_clip=True,
        score=92.0,
        hook_score=90.0,
        payoff_score=94.0,
        self_contained_score=92.0,
        reason="Excellent clip",
        suggested_start=10.0,
        suggested_end=45.0,
    )
    mocks["semantic_scorer"].score_candidates.return_value = [score]
    mocks["semantic_scorer"].select_best_clip.return_value = (candidate, score, "APPROVED")

    mocks["boundary_refiner"].refine.return_value = RefinementResult(
        is_valid=True, refined_start=10.0, refined_end=45.0, duration=35.0
    )

    mocks["visual_analyzer"].analyze_clip.return_value = VisualAnalysisReport(
        video_path=str(dummy_video_file),
        start_sec=10.0,
        end_sec=45.0,
        duration=35.0,
        scene_cuts=[20.0],
        subject_presence_ratio=0.9,
    )
    mocks["visual_director"].evaluate_window.return_value = VisualDirectorVerdict(
        approved=True, confidence=0.96, shot_type="single_speaker", notes="Viable visuals"
    )
    mocks["framing"].analyze_framing.return_value = FramingDecision(
        layout="PORTRAIT_9_16",
        crop_windows=[SceneCrop(scene_start=10.0, scene_end=45.0, crop_x=420, crop_y=0, crop_w=1080, crop_h=1920)],
    )
    mocks["subtitle_classifier"].check_burned_in_subtitles.return_value = (False, None)

    def fake_render(video_path, output_path, edit_plan, start_sec, duration, subtitle_ass_path=None):
        Path(output_path).write_bytes(b"\x00" * (110 * 1024))
        return RenderResult(output_path=str(output_path), duration=duration, success=True)
    mocks["renderer"].render.side_effect = fake_render

    mocks["qc_gate"].evaluate.return_value = ThreeTierQCReport(
        passed=True,
        publishable=True,
        technical=TechnicalQCResult(passed=True, duration=35.0, width=1080, height=1920, video_codec="h264", audio_codec="aac", sample_rate=48000),
        visual=VisualQCResult(passed=True, sampled_frames_count=18, blank_frames=0, subject_present_ratio=0.9, subtitle_safe=True),
        perceptual=PerceptualQCResult(passed=True, publishable=True, score=90, blocking_issues=[]),
    )

    mocks["uploader"].upload_short.return_value = {
        "status": "success",
        "platform": "youtube",
        "video_id": "dry_run_99",
        "url": "https://youtube.com/shorts/dry_run_99",
    }

    # Execute process_video using VideoSourceMeta directly
    result = orch.process_video(video_meta_or_url=vid_meta, custom_video_path=str(dummy_video_file), dry_run=True)

    assert result.is_success is True
    assert result.status == PipelineStatus.COMPLETED
    assert result.video_id == "happy_vid_99"
    assert result.rendered_path is not None
    assert result.duration_sec == 35.0
    assert result.gate_check is not None
    assert result.gate_check.passed is True

    # Verify DB persistence
    db_v = temp_db.get_video("happy_vid_99")
    assert db_v is not None
    assert db_v.status == PipelineStatus.COMPLETED


# ==============================================================================
# 4. Reject Is a Valid Outcome Tests (Stage Rejections)
# ==============================================================================

def test_reject_outcome_source_filter(tmp_path, temp_db, dummy_video_file):
    """SourceFilter rejection transitions to rejected_language without crashing."""
    orch, mocks = _build_mock_orchestrator_harness(tmp_path, temp_db, dummy_video_file)

    meta = VideoSourceMeta(video_id="rej_sf", url="http://v", title="Short music", duration_sec=40.0)
    mocks["searcher"].get_video_metadata.return_value = meta
    mocks["source_filter"].filter_video.return_value = EligibilityResult(
        is_eligible=False, rejection_code="DURATION_TOO_SHORT", reason="Video duration < 180s"
    )

    result = orch.process_video(video_meta_or_url=meta)

    assert result.is_success is False
    assert result.status == PipelineStatus.REJECTED_LANGUAGE
    assert "DURATION_TOO_SHORT" in (result.rejection_reason or "")

    db_v = temp_db.get_video("rej_sf")
    assert db_v is not None
    assert db_v.status == PipelineStatus.REJECTED_LANGUAGE


def test_reject_outcome_language_gate(tmp_path, temp_db, dummy_video_file):
    """LanguageGate rejection transitions to rejected_language without crashing."""
    orch, mocks = _build_mock_orchestrator_harness(tmp_path, temp_db, dummy_video_file)

    meta = VideoSourceMeta(video_id="rej_lang", url="http://v", title="English Tech Talk", duration_sec=400.0)
    mocks["searcher"].get_video_metadata.return_value = meta
    mocks["source_filter"].filter_video.return_value = EligibilityResult(is_eligible=True, reason="OK")

    mocks["transcript_provider"].get_phrase_transcript.return_value = [
        TranscriptSegment(start=0.0, end=20.0, duration=20.0, text="Today we will discuss distributed architecture systems.")
    ]
    mocks["language_gate"].evaluate_transcript.return_value = LanguageGateResult(
        eligible=False, primary_language="en", confidence=0.98, reason="English-dominant audio"
    )

    result = orch.process_video(video_meta_or_url=meta, custom_video_path=str(dummy_video_file))

    assert result.is_success is False
    assert result.status == PipelineStatus.REJECTED_LANGUAGE
    assert "Indonesian Language Gate REJECTED" in (result.rejection_reason or "")

    db_v = temp_db.get_video("rej_lang")
    assert db_v is not None
    assert db_v.status == PipelineStatus.REJECTED_LANGUAGE


def test_reject_outcome_no_candidates(tmp_path, temp_db, dummy_video_file):
    """CandidateGenerator producing 0 candidates transitions to no_good_clip."""
    orch, mocks = _build_mock_orchestrator_harness(tmp_path, temp_db, dummy_video_file)

    meta = VideoSourceMeta(video_id="rej_cand", url="http://v", title="Hening", duration_sec=300.0)
    mocks["searcher"].get_video_metadata.return_value = meta
    mocks["source_filter"].filter_video.return_value = EligibilityResult(is_eligible=True, reason="OK")
    mocks["transcript_provider"].get_phrase_transcript.return_value = [
        TranscriptSegment(start=0.0, end=5.0, duration=5.0, text="Halo.")
    ]
    mocks["language_gate"].evaluate_transcript.return_value = LanguageGateResult(eligible=True, primary_language="id", confidence=0.9, reason="OK")
    mocks["candidate_generator"].generate_candidates.return_value = []

    result = orch.process_video(video_meta_or_url=meta, custom_video_path=str(dummy_video_file))

    assert result.is_success is False
    assert result.status == PipelineStatus.NO_GOOD_CLIP
    assert "0 viable candidate windows" in (result.rejection_reason or "")

    db_v = temp_db.get_video("rej_cand")
    assert db_v is not None
    assert db_v.status == PipelineStatus.NO_GOOD_CLIP


def test_reject_outcome_semantic_scorer(tmp_path, temp_db, dummy_video_file):
    """SemanticScorer rejecting candidate transitions to no_good_clip."""
    orch, mocks = _build_mock_orchestrator_harness(tmp_path, temp_db, dummy_video_file)

    meta = VideoSourceMeta(video_id="rej_sem", url="http://v", title="Obrolan Biasa", duration_sec=300.0)
    mocks["searcher"].get_video_metadata.return_value = meta
    mocks["source_filter"].filter_video.return_value = EligibilityResult(is_eligible=True, reason="OK")
    mocks["transcript_provider"].get_phrase_transcript.return_value = [
        TranscriptSegment(start=0.0, end=35.0, duration=35.0, text="Ini obrolan biasa tanpa hook.")
    ]
    mocks["language_gate"].evaluate_transcript.return_value = LanguageGateResult(eligible=True, primary_language="id", confidence=0.9, reason="OK")

    c = CandidateWindow(candidate_id="c_boring", start_sec=0.0, end_sec=35.0, duration_sec=35.0, text="...")
    mocks["candidate_generator"].generate_candidates.return_value = [c]
    mocks["semantic_scorer"].score_candidates.return_value = [
        SemanticScore(
            candidate_id="c_boring",
            good_clip=False,
            score=40.0,
            hook_score=30.0,
            payoff_score=30.0,
            self_contained_score=30.0,
            reason="No hook or payoff",
            suggested_start=0.0,
            suggested_end=35.0,
        )
    ]
    mocks["semantic_scorer"].select_best_clip.return_value = (None, None, "NO_GOOD_CLIP_FOUND")

    result = orch.process_video(video_meta_or_url=meta, custom_video_path=str(dummy_video_file))

    assert result.is_success is False
    assert result.status == PipelineStatus.NO_GOOD_CLIP
    assert "NO_GOOD_CLIP_FOUND" in (result.rejection_reason or "")

    db_v = temp_db.get_video("rej_sem")
    assert db_v is not None
    assert db_v.status == PipelineStatus.NO_GOOD_CLIP


def test_reject_outcome_boundary_refiner(tmp_path, temp_db, dummy_video_file):
    """BoundaryRefiner rejecting boundary window transitions to no_good_clip."""
    orch, mocks = _build_mock_orchestrator_harness(tmp_path, temp_db, dummy_video_file)

    meta = VideoSourceMeta(video_id="rej_bound", url="http://v", title="Boundary Cut", duration_sec=300.0)
    mocks["searcher"].get_video_metadata.return_value = meta
    mocks["source_filter"].filter_video.return_value = EligibilityResult(is_eligible=True, reason="OK")
    mocks["transcript_provider"].get_phrase_transcript.return_value = [
        TranscriptSegment(start=0.0, end=35.0, duration=35.0, text="Obrolan terpotong.")
    ]
    mocks["language_gate"].evaluate_transcript.return_value = LanguageGateResult(eligible=True, primary_language="id", confidence=0.9, reason="OK")

    c = CandidateWindow(candidate_id="c_bound", start_sec=0.0, end_sec=35.0, duration_sec=35.0, text="...")
    s = SemanticScore(
        candidate_id="c_bound",
        good_clip=True,
        score=80.0,
        hook_score=80.0,
        payoff_score=80.0,
        self_contained_score=80.0,
        reason="Good",
        suggested_start=0.0,
        suggested_end=35.0,
    )
    mocks["candidate_generator"].generate_candidates.return_value = [c]
    mocks["semantic_scorer"].score_candidates.return_value = [s]
    mocks["semantic_scorer"].select_best_clip.return_value = (c, s, "APPROVED")

    mocks["boundary_refiner"].refine.return_value = RefinementResult(
        is_valid=False,
        refined_start=0.0,
        refined_end=22.0,
        rejection_reason="Cannot find clean speech boundary",
        duration=22.0,
    )

    result = orch.process_video(video_meta_or_url=meta, custom_video_path=str(dummy_video_file))

    assert result.is_success is False
    assert result.status == PipelineStatus.NO_GOOD_CLIP
    assert "failed boundary refinement" in (result.rejection_reason or "") or "Boundary Refiner REJECTED" in (result.rejection_reason or "")

    db_v = temp_db.get_video("rej_bound")
    assert db_v is not None
    assert db_v.status == PipelineStatus.NO_GOOD_CLIP


def test_reject_outcome_visual_director(tmp_path, temp_db, dummy_video_file):
    """VisualDirector rejection transitions to rejected_visual."""
    orch, mocks = _build_mock_orchestrator_harness(tmp_path, temp_db, dummy_video_file)

    meta = VideoSourceMeta(video_id="rej_vis", url="http://v", title="Layar Gelap", duration_sec=300.0)
    mocks["searcher"].get_video_metadata.return_value = meta
    mocks["source_filter"].filter_video.return_value = EligibilityResult(is_eligible=True, reason="OK")
    mocks["transcript_provider"].get_phrase_transcript.return_value = [
        TranscriptSegment(start=0.0, end=35.0, duration=35.0, text="Layar hitam doang.")
    ]
    mocks["language_gate"].evaluate_transcript.return_value = LanguageGateResult(eligible=True, primary_language="id", confidence=0.9, reason="OK")

    c = CandidateWindow(candidate_id="c_vis", start_sec=0.0, end_sec=35.0, duration_sec=35.0, text="...")
    s = SemanticScore(
        candidate_id="c_vis",
        good_clip=True,
        score=80.0,
        hook_score=80.0,
        payoff_score=80.0,
        self_contained_score=80.0,
        reason="Good",
        suggested_start=0.0,
        suggested_end=35.0,
    )
    mocks["candidate_generator"].generate_candidates.return_value = [c]
    mocks["semantic_scorer"].score_candidates.return_value = [s]
    mocks["semantic_scorer"].select_best_clip.return_value = (c, s, "APPROVED")
    mocks["boundary_refiner"].refine.return_value = RefinementResult(is_valid=True, refined_start=0.0, refined_end=35.0, duration=35.0)

    mocks["visual_analyzer"].analyze_clip.return_value = VisualAnalysisReport(
        video_path=str(dummy_video_file), start_sec=0.0, end_sec=35.0, duration=35.0, scene_cuts=[], subject_presence_ratio=0.1
    )
    mocks["visual_director"].evaluate_window.return_value = VisualDirectorVerdict(
        approved=False, rejection_reasons=["No clear subject visible across clip"]
    )

    result = orch.process_video(video_meta_or_url=meta, custom_video_path=str(dummy_video_file))

    assert result.is_success is False
    assert result.status == PipelineStatus.REJECTED_VISUAL
    assert "No clear subject visible" in (result.rejection_reason or "")

    db_v = temp_db.get_video("rej_vis")
    assert db_v is not None
    assert db_v.status == PipelineStatus.REJECTED_VISUAL


def test_reject_outcome_renderer_failure(tmp_path, temp_db, dummy_video_file):
    """FFmpeg CleanRenderer failure transitions to render_failed."""
    orch, mocks = _build_mock_orchestrator_harness(tmp_path, temp_db, dummy_video_file)

    meta = VideoSourceMeta(video_id="rej_ren", url="http://v", title="Render Corrupt", duration_sec=300.0)
    mocks["searcher"].get_video_metadata.return_value = meta
    mocks["source_filter"].filter_video.return_value = EligibilityResult(is_eligible=True, reason="OK")
    mocks["transcript_provider"].get_phrase_transcript.return_value = [
        TranscriptSegment(start=0.0, end=35.0, duration=35.0, text="Cerita.")
    ]
    mocks["language_gate"].evaluate_transcript.return_value = LanguageGateResult(eligible=True, primary_language="id", confidence=0.9, reason="OK")

    c = CandidateWindow(candidate_id="c_ren", start_sec=0.0, end_sec=35.0, duration_sec=35.0, text="...")
    s = SemanticScore(
        candidate_id="c_ren",
        good_clip=True,
        score=80.0,
        hook_score=80.0,
        payoff_score=80.0,
        self_contained_score=80.0,
        reason="Good",
        suggested_start=0.0,
        suggested_end=35.0,
    )
    mocks["candidate_generator"].generate_candidates.return_value = [c]
    mocks["semantic_scorer"].score_candidates.return_value = [s]
    mocks["semantic_scorer"].select_best_clip.return_value = (c, s, "APPROVED")
    mocks["boundary_refiner"].refine.return_value = RefinementResult(is_valid=True, refined_start=0.0, refined_end=35.0, duration=35.0)

    mocks["visual_analyzer"].analyze_clip.return_value = VisualAnalysisReport(
        video_path=str(dummy_video_file), start_sec=0.0, end_sec=35.0, duration=35.0, scene_cuts=[]
    )
    mocks["visual_director"].evaluate_window.return_value = VisualDirectorVerdict(approved=True, shot_type="single_speaker")
    mocks["framing"].analyze_framing.return_value = FramingDecision(layout="PORTRAIT_9_16", crop_windows=[SceneCrop(scene_start=0.0, scene_end=35.0, crop_x=0, crop_y=0, crop_w=1080, crop_h=1920)])
    mocks["subtitle_classifier"].check_burned_in_subtitles.return_value = (False, None)

    # Renderer fails
    mocks["renderer"].render.return_value = RenderResult(output_path="", duration=0.0, success=False, error_message="Encoder segfault")

    result = orch.process_video(video_meta_or_url=meta, custom_video_path=str(dummy_video_file))

    assert result.is_success is False
    assert result.status == PipelineStatus.RENDER_FAILED
    assert "Encoder segfault" in (result.error_message or "")

    db_v = temp_db.get_video("rej_ren")
    assert db_v is not None
    assert db_v.status == PipelineStatus.RENDER_FAILED


def test_reject_outcome_qc_failure(tmp_path, temp_db, dummy_video_file):
    """Three-Tier QC rejection transitions to qc_failed."""
    orch, mocks = _build_mock_orchestrator_harness(tmp_path, temp_db, dummy_video_file)

    meta = VideoSourceMeta(video_id="rej_qc", url="http://v", title="QC Defect", duration_sec=300.0)
    mocks["searcher"].get_video_metadata.return_value = meta
    mocks["source_filter"].filter_video.return_value = EligibilityResult(is_eligible=True, reason="OK")
    mocks["transcript_provider"].get_phrase_transcript.return_value = [
        TranscriptSegment(start=0.0, end=35.0, duration=35.0, text="Cerita QC.")
    ]
    mocks["language_gate"].evaluate_transcript.return_value = LanguageGateResult(eligible=True, primary_language="id", confidence=0.9, reason="OK")

    c = CandidateWindow(candidate_id="c_qc", start_sec=0.0, end_sec=35.0, duration_sec=35.0, text="...")
    s = SemanticScore(
        candidate_id="c_qc",
        good_clip=True,
        score=80.0,
        hook_score=80.0,
        payoff_score=80.0,
        self_contained_score=80.0,
        reason="Good",
        suggested_start=0.0,
        suggested_end=35.0,
    )
    mocks["candidate_generator"].generate_candidates.return_value = [c]
    mocks["semantic_scorer"].score_candidates.return_value = [s]
    mocks["semantic_scorer"].select_best_clip.return_value = (c, s, "APPROVED")
    mocks["boundary_refiner"].refine.return_value = RefinementResult(is_valid=True, refined_start=0.0, refined_end=35.0, duration=35.0)

    mocks["visual_analyzer"].analyze_clip.return_value = VisualAnalysisReport(
        video_path=str(dummy_video_file), start_sec=0.0, end_sec=35.0, duration=35.0, scene_cuts=[]
    )
    mocks["visual_director"].evaluate_window.return_value = VisualDirectorVerdict(approved=True, shot_type="single_speaker")
    mocks["framing"].analyze_framing.return_value = FramingDecision(layout="PORTRAIT_9_16", crop_windows=[SceneCrop(scene_start=0.0, scene_end=35.0, crop_x=0, crop_y=0, crop_w=1080, crop_h=1920)])
    mocks["subtitle_classifier"].check_burned_in_subtitles.return_value = (False, None)

    def fake_render(video_path, output_path, edit_plan, start_sec, duration, subtitle_ass_path=None):
        Path(output_path).write_bytes(b"\x00" * (110 * 1024))
        return RenderResult(output_path=str(output_path), duration=duration, success=True)
    mocks["renderer"].render.side_effect = fake_render

    # QC rejects due to technical audio failure
    mocks["qc_gate"].evaluate.return_value = ThreeTierQCReport(
        passed=False,
        publishable=False,
        errors=["Audio codec mismatch: mp3 != aac"],
        technical=TechnicalQCResult(passed=False, duration=35.0, width=1080, height=1920, video_codec="h264", audio_codec="mp3", sample_rate=48000, errors=["audio_codec_mismatch"]),
        visual=VisualQCResult(passed=True, sampled_frames_count=18, blank_frames=0, subject_present_ratio=0.9, subtitle_safe=True),
        perceptual=PerceptualQCResult(passed=True, publishable=True, score=85),
    )

    result = orch.process_video(video_meta_or_url=meta, custom_video_path=str(dummy_video_file))

    assert result.is_success is False
    assert result.status == PipelineStatus.QC_FAILED
    assert "Three-Tier QC Gate REJECTED" in (result.error_message or "")

    db_v = temp_db.get_video("rej_qc")
    assert db_v is not None
    assert db_v.status == PipelineStatus.QC_FAILED


def test_reject_outcome_upload_failure(tmp_path, temp_db, dummy_video_file):
    """Platform upload failure transitions to upload_failed."""
    orch, mocks = _build_mock_orchestrator_harness(tmp_path, temp_db, dummy_video_file)

    meta = VideoSourceMeta(video_id="rej_up", url="http://v", title="Upload Error", duration_sec=300.0)
    mocks["searcher"].get_video_metadata.return_value = meta
    mocks["source_filter"].filter_video.return_value = EligibilityResult(is_eligible=True, reason="OK")
    mocks["transcript_provider"].get_phrase_transcript.return_value = [
        TranscriptSegment(start=0.0, end=35.0, duration=35.0, text="Cerita upload.")
    ]
    mocks["language_gate"].evaluate_transcript.return_value = LanguageGateResult(eligible=True, primary_language="id", confidence=0.9, reason="OK")

    c = CandidateWindow(candidate_id="c_up", start_sec=0.0, end_sec=35.0, duration_sec=35.0, text="...")
    s = SemanticScore(
        candidate_id="c_up",
        good_clip=True,
        score=80.0,
        hook_score=80.0,
        payoff_score=80.0,
        self_contained_score=80.0,
        reason="Good",
        suggested_start=0.0,
        suggested_end=35.0,
    )
    mocks["candidate_generator"].generate_candidates.return_value = [c]
    mocks["semantic_scorer"].score_candidates.return_value = [s]
    mocks["semantic_scorer"].select_best_clip.return_value = (c, s, "APPROVED")
    mocks["boundary_refiner"].refine.return_value = RefinementResult(is_valid=True, refined_start=0.0, refined_end=35.0, duration=35.0)

    mocks["visual_analyzer"].analyze_clip.return_value = VisualAnalysisReport(
        video_path=str(dummy_video_file), start_sec=0.0, end_sec=35.0, duration=35.0, scene_cuts=[]
    )
    mocks["visual_director"].evaluate_window.return_value = VisualDirectorVerdict(approved=True, shot_type="single_speaker")
    mocks["framing"].analyze_framing.return_value = FramingDecision(layout="PORTRAIT_9_16", crop_windows=[SceneCrop(scene_start=0.0, scene_end=35.0, crop_x=0, crop_y=0, crop_w=1080, crop_h=1920)])
    mocks["subtitle_classifier"].check_burned_in_subtitles.return_value = (False, None)

    def fake_render(video_path, output_path, edit_plan, start_sec, duration, subtitle_ass_path=None):
        Path(output_path).write_bytes(b"\x00" * (110 * 1024))
        return RenderResult(output_path=str(output_path), duration=duration, success=True)
    mocks["renderer"].render.side_effect = fake_render

    mocks["qc_gate"].evaluate.return_value = ThreeTierQCReport(
        passed=True,
        publishable=True,
        technical=TechnicalQCResult(passed=True, duration=35.0, width=1080, height=1920, video_codec="h264", audio_codec="aac", sample_rate=48000),
        visual=VisualQCResult(passed=True, sampled_frames_count=18, blank_frames=0, subject_present_ratio=0.9, subtitle_safe=True),
        perceptual=PerceptualQCResult(passed=True, publishable=True, score=90),
    )

    # Uploader fails
    mocks["uploader"].upload_short.return_value = {
        "status": "failed",
        "error": "YouTube quotaExceeded",
    }

    result = orch.process_video(video_meta_or_url=meta, custom_video_path=str(dummy_video_file))

    assert result.is_success is False
    assert result.status == PipelineStatus.UPLOAD_FAILED
    assert "quotaExceeded" in (result.error_message or "")

    db_v = temp_db.get_video("rej_up")
    assert db_v is not None
    assert db_v.status == PipelineStatus.UPLOAD_FAILED


# ==============================================================================
# 5. Discovery Cycle Tests (Bab 22)
# ==============================================================================

def test_run_discovery_cycle_multi_query(temp_db):
    """run_discovery_cycle supports string or list of queries and handles mixed outcomes."""
    mock_searcher = MagicMock()
    mock_searcher.search_eligible_videos.side_effect = [
        [
            VideoSourceMeta(video_id="v_q1_a", url="http://v1a", title="Podcast 1", duration_sec=300.0),
            VideoSourceMeta(video_id="v_q1_b", url="http://v1b", title="Podcast 2", duration_sec=300.0),
        ],
        [
            VideoSourceMeta(video_id="v_q2_a", url="http://v2a", title="Standup 1", duration_sec=300.0),
            VideoSourceMeta(video_id="v_q1_a", url="http://v1a", title="Duplicate", duration_sec=300.0),  # Deduplicate
        ],
    ]

    orch = AutoClipperOrchestrator(repository=temp_db, searcher=mock_searcher)

    with patch.object(orch, "process_video") as mock_pv:
        mock_pv.side_effect = [
            PipelineResult(video_id="v_q1_a", status=PipelineStatus.COMPLETED, is_success=True),
            PipelineResult(video_id="v_q1_b", status=PipelineStatus.NO_GOOD_CLIP, is_success=False, rejection_reason="No hook"),
            PipelineResult(video_id="v_q2_a", status=PipelineStatus.COMPLETED, is_success=True),
        ]

        results = orch.run_discovery_cycle(
            search_queries=["podcast viral", "standup comedy"],
            max_videos=3,
            dry_run=True,
        )

        assert len(results) == 3
        assert results[0].video_id == "v_q1_a"
        assert results[0].is_success is True
        assert results[1].video_id == "v_q1_b"
        assert results[1].is_success is False
        assert results[2].video_id == "v_q2_a"
        assert results[2].is_success is True


def test_run_discovery_cycle_single_query_legacy_kwarg(temp_db):
    """run_discovery_cycle supports legacy query='...' kwarg."""
    mock_searcher = MagicMock()
    mock_searcher.search_eligible_videos.return_value = [
        VideoSourceMeta(video_id="v_legacy", url="http://vl", title="Legacy", duration_sec=300.0)
    ]

    orch = AutoClipperOrchestrator(repository=temp_db, searcher=mock_searcher)

    with patch.object(orch, "process_video") as mock_pv:
        mock_pv.return_value = PipelineResult(video_id="v_legacy", status=PipelineStatus.COMPLETED, is_success=True)

        results = orch.run_discovery_cycle(query="podcast trending", max_videos=1)
        assert len(results) == 1
        assert results[0].video_id == "v_legacy"
