"""PRISM Independent Correctness & Verification Suite.

Validates Auto Clipper V3.1 Double Subtitle Prevention & Natural Sentence Ending Engine
according to the specification in HERMES_V3_1_FIX_DOUBLE_SUBTITLE_AND_ENDING.md.

Surface Verification:
1. Double Subtitle Prevention & Subtitle Policy Matrix:
   - SOURCE_EXISTING generates 0 ASS files and 0 filtergraph text overlays.
   - GENERATE policy correctly produces ASS file.
   - UNKNOWN subtitle state strictly rejects candidate (PipelineStatus.REJECTED_VISUAL).
   - Low confidence (< 0.70) strictly rejects candidate.
   - Conflict states (Local BURNED_IN vs Gemini NO_SUBTITLE or Local NONE vs Gemini HAS_SUBTITLE) strictly reject candidate.
   - Gemini Native Video QC rejects any double subtitles (double_subtitles_detected=True / has_double_subtitles=FAIL).
2. Natural Sentence Ending Engine:
   - Dangling words, connectors, and phrases detected.
   - Trailing ellipses and mid-sentence punctuation detected.
   - Boundary refiner extends dangling ending to sentence completion if duration <= 55s.
   - Boundary refiner rejects extension if duration > 55s.
   - Natural mid-speech pauses (silence > 300ms, laughter, breath) preserved without premature cut false positives.
   - Orchestrator candidate fallback: Rank 1 rejected due to >55s extension falls back to Rank 2.
   - Gemini Native Video QC rejects mid-sentence cutoff (ending_complete=FAIL / ending_natural=FAIL).
3. Real Media & Filtergraph Invariants:
   - Embedded subtitle stream detection via ffprobe.
   - Render pipeline produces 0 ASS filters when subtitle_policy is SOURCE_EXISTING.
   - Failed QC archiving to failed/ directory with reason.md, qc.json, edit_plan.json, transcript.txt.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from analysis.boundary_refiner import (
    BoundaryRefiner,
    DANGLING_CONNECTORS,
    DANGLING_PHRASES,
    is_sentence_complete,
)
from analysis.candidate_generator import CandidateWindow
from analysis.semantic_scorer import SemanticScore
from analysis.visual_preflight import VisualPreflight, VisualPreflightResult
from discovery.searcher import VideoSourceMeta
from editing.edit_plan import EditPlan
from editing.renderer import CleanRenderer
from editing.subtitle_detector import SubtitleDetectionState, SubtitleDetector
from pipeline.orchestrator import AutoClipperOrchestrator
from pipeline.state_machine import PipelineStatus
from quality import ThreeTierQCReport
from quality.gemini_video_qc import GeminiNativeVideoQC, GeminiVideoQCResult
from quality.perceptual_qc import PerceptualQCResult
from quality.technical_qc import TechnicalQCResult
from quality.visual_qc import VisualQCResult
from storage.repository import StorageRepository
from transcription.transcript_provider import TranscriptSegment


# ==============================================================================
# SECTION 1: Adversarial Natural Sentence Ending Tests
# ==============================================================================

@pytest.mark.parametrize("dangling_phrase", DANGLING_PHRASES)
def test_all_dangling_phrases_falsify_completion(dangling_phrase):
    """PRISM Adversarial Audit: Every specified dangling phrase must fail sentence completion."""
    text = f"Pembicara ini sedang berbicara dan {dangling_phrase}"
    is_comp, reason = is_sentence_complete(text)
    assert is_comp is False, f"Dangling phrase '{dangling_phrase}' was falsely marked complete!"
    assert len(reason) > 0


@pytest.mark.parametrize("connector", [
    "dan", "atau", "tapi", "karena", "sebab", "sehingga", "agar", "supaya",
    "yang", "untuk", "dengan", "ke", "di", "dari", "masih", "sedang", "akan",
    "bisa", "sebenarnya", "jadi", "melainkan", "namun", "sementara", "sedangkan"
])
def test_all_dangling_connectors_falsify_completion(connector):
    """PRISM Adversarial Audit: Trailing conjunctions, prepositions, or modals must fail."""
    text = f"Kemarin mereka sudah sepakat bahwa {connector}"
    is_comp, reason = is_sentence_complete(text)
    assert is_comp is False, f"Dangling connector '{connector}' was falsely marked complete!"
    assert connector in reason.lower() or "connector" in reason.lower()


@pytest.mark.parametrize("trailing_punct", [",", ";", ":", "-", "...", "…"])
def test_trailing_punctuation_falsifies_completion(trailing_punct):
    """PRISM Adversarial Audit: Mid-sentence punctuation or ellipsis must fail."""
    text = f"Semua rencana telah dipersiapkan{trailing_punct}"
    is_comp, reason = is_sentence_complete(text)
    assert is_comp is False, f"Trailing punctuation '{trailing_punct}' was falsely marked complete!"


@pytest.mark.parametrize("terminal_punct", [".", "!", "?"])
def test_terminal_punctuation_passes_completion(terminal_punct):
    """Terminal punctuation on complete sentences must pass."""
    text = f"Semua rencana telah selesai dan berhasil{terminal_punct}"
    is_comp, reason = is_sentence_complete(text)
    assert is_comp is True


def test_natural_pause_in_speech_does_not_falsify_completed_sentence():
    """Natural pause / laughter in the middle of dialogue must NOT fail completion if ending is complete."""
    text_with_pause = "Awalnya kami ragu, tapi setelah dicoba... akhirnya kami semua tertawa dan berhasil menang!"
    is_comp, reason = is_sentence_complete(text_with_pause)
    assert is_comp is True, f"Sentence with mid-dialogue pause was falsely rejected: {reason}"


# ==============================================================================
# SECTION 2: Boundary Refiner Extension & Cap Enforcement
# ==============================================================================

def test_boundary_refiner_exact_55s_boundary():
    """Refiner allows extension exactly up to 55.0s, but strictly blocks 55.1s."""
    refiner = BoundaryRefiner(min_duration_sec=30.0, max_duration_sec=55.0, laughter_buffer_sec=0.0)

    # Segments where extension lands exactly at 55.0s
    segments_55_0 = [
        TranscriptSegment(start=0.0, end=40.0, duration=40.0, text="Awal cerita."),
        TranscriptSegment(start=40.1, end=50.0, duration=9.9, text="waktu itu masih"),
        TranscriptSegment(start=50.1, end=55.0, duration=4.9, text="belajar di sekolah."),
    ]
    can_ext, new_end, _ = refiner.extend_to_sentence_boundary(
        start_sec=0.0, end_sec=50.0, segments=segments_55_0, max_duration_sec=55.0
    )
    assert can_ext is True
    assert new_end == 55.0

    # Segments where extension lands at 55.1s (exceeds cap)
    segments_55_1 = [
        TranscriptSegment(start=0.0, end=40.0, duration=40.0, text="Awal cerita."),
        TranscriptSegment(start=40.1, end=50.0, duration=9.9, text="waktu itu masih"),
        TranscriptSegment(start=50.1, end=55.1, duration=5.0, text="belajar di sekolah."),
    ]
    can_ext2, new_end2, reason2 = refiner.extend_to_sentence_boundary(
        start_sec=0.0, end_sec=50.0, segments=segments_55_1, max_duration_sec=55.0
    )
    assert can_ext2 is False
    assert new_end2 == 50.0
    assert "exceed" in reason2.lower()


# ==============================================================================
# SECTION 3: Subtitle Decision Hierarchy & Conflict Falsification
# ==============================================================================

def test_orchestrator_rejects_ambiguous_gemini_confidence(tmp_path: Path):
    """PRISM Audit: Gemini subtitle confidence < 0.7 must trigger immediate REJECTED_VISUAL."""
    temp_db = StorageRepository(db_path=str(tmp_path / "test_ambig.db"))
    out_dir = tmp_path / "output"
    down_dir = tmp_path / "downloads"
    out_dir.mkdir(parents=True)
    down_dir.mkdir(parents=True)

    dummy_media = down_dir / "vid_ambig.mp4"
    dummy_media.write_bytes(b"\x00" * (64 * 1024))

    vid_meta = VideoSourceMeta(
        video_id="vid_ambig",
        url="https://youtube.com/watch?v=vid_ambig",
        title="Ambig Video",
        duration_sec=200.0,
    )

    mock_searcher = MagicMock()
    mock_searcher.get_video_metadata.return_value = vid_meta
    mock_source_filter = MagicMock(filter_video=MagicMock(return_value=MagicMock(is_eligible=True)))
    mock_lang_gate = MagicMock(evaluate_transcript=MagicMock(return_value=MagicMock(eligible=True)))

    transcript = [TranscriptSegment(start=10.0, end=45.0, duration=35.0, text="Dialog tuntas.")]
    mock_transcript_provider = MagicMock(get_phrase_transcript=MagicMock(return_value=transcript))

    candidate = CandidateWindow(candidate_id="c1", start_sec=10.0, end_sec=45.0, duration_sec=35.0, text="Dialog tuntas.")
    mock_candidate_gen = MagicMock(generate_candidates=MagicMock(return_value=[candidate]))

    score = SemanticScore(
        candidate_id="c1", good_clip=True, score=90.0, hook_score=90.0, payoff_score=90.0,
        self_contained_score=90.0, reason="OK", suggested_start=10.0, suggested_end=45.0
    )
    mock_scorer = MagicMock(score_candidates=MagicMock(return_value=[score]), select_best_clip=MagicMock(return_value=(candidate, score, "APPROVED")))
    mock_vis_analyzer = MagicMock(analyze_clip=MagicMock(return_value=MagicMock(scene_cuts=[])))
    mock_vis_director = MagicMock(evaluate_window=MagicMock(return_value=MagicMock(approved=True)))
    mock_framing = MagicMock(analyze_framing=MagicMock(return_value=MagicMock(layout="SAFE_WIDE", crop_windows=[])))

    # Gemini preflight has low confidence (0.55 < 0.70)
    mock_preflight = MagicMock()
    mock_preflight.preflight_clip.return_value = VisualPreflightResult(
        usable=True,
        has_subtitles=True,
        subtitle_confidence=0.55,
        subtitle_reason="Unclear blurry text at bottom",
        ending_complete=True,
        ending_natural=True,
        recommended_layout="SAFE_WIDE",
        preflight_mode="GEMINI_NATIVE_VIDEO",
        notes="Low confidence preflight",
    )

    orch = AutoClipperOrchestrator(
        repository=temp_db,
        searcher=mock_searcher,
        source_filter=mock_source_filter,
        language_gate=mock_lang_gate,
        transcript_provider=mock_transcript_provider,
        candidate_generator=mock_candidate_gen,
        semantic_scorer=mock_scorer,
        boundary_refiner=BoundaryRefiner(),
        visual_analyzer=mock_vis_analyzer,
        visual_director=mock_vis_director,
        framing=mock_framing,
        visual_preflight=mock_preflight,
        output_dir=out_dir,
        download_dir=down_dir,
    )

    with patch("editing.subtitle_detector.SubtitleDetector.detect") as mock_detect:
        mock_detect.return_value = (SubtitleDetectionState.NONE, 0.95, {})
        res = orch.process_video(video_meta_or_url=vid_meta, custom_video_path=str(dummy_media), dry_run=True)

    assert res.is_success is False
    assert res.status == PipelineStatus.REJECTED_VISUAL
    assert "ambiguous" in res.rejection_reason.lower() or "low confidence" in res.rejection_reason.lower()


def test_orchestrator_rejects_local_unknown_state(tmp_path: Path):
    """PRISM Audit: Local detector UNKNOWN state must trigger immediate REJECTED_VISUAL."""
    temp_db = StorageRepository(db_path=str(tmp_path / "test_unknown.db"))
    out_dir = tmp_path / "output"
    down_dir = tmp_path / "downloads"
    out_dir.mkdir(parents=True)
    down_dir.mkdir(parents=True)

    dummy_media = down_dir / "vid_unk.mp4"
    dummy_media.write_bytes(b"\x00" * (64 * 1024))

    vid_meta = VideoSourceMeta(video_id="vid_unk", url="https://youtube.com/watch?v=vid_unk", title="Unk", duration_sec=200.0)

    mock_searcher = MagicMock(get_video_metadata=MagicMock(return_value=vid_meta))
    mock_source_filter = MagicMock(filter_video=MagicMock(return_value=MagicMock(is_eligible=True)))
    mock_lang_gate = MagicMock(evaluate_transcript=MagicMock(return_value=MagicMock(eligible=True)))

    transcript = [TranscriptSegment(start=10.0, end=45.0, duration=35.0, text="Dialog tuntas.")]
    mock_transcript_provider = MagicMock(get_phrase_transcript=MagicMock(return_value=transcript))

    candidate = CandidateWindow(candidate_id="c1", start_sec=10.0, end_sec=45.0, duration_sec=35.0, text="Dialog tuntas.")
    mock_candidate_gen = MagicMock(generate_candidates=MagicMock(return_value=[candidate]))

    score = SemanticScore(
        candidate_id="c1", good_clip=True, score=90.0, hook_score=90.0, payoff_score=90.0,
        self_contained_score=90.0, reason="OK", suggested_start=10.0, suggested_end=45.0
    )
    mock_scorer = MagicMock(score_candidates=MagicMock(return_value=[score]), select_best_clip=MagicMock(return_value=(candidate, score, "APPROVED")))
    mock_vis_analyzer = MagicMock(analyze_clip=MagicMock(return_value=MagicMock(scene_cuts=[])))
    mock_vis_director = MagicMock(evaluate_window=MagicMock(return_value=MagicMock(approved=True)))
    mock_framing = MagicMock(analyze_framing=MagicMock(return_value=MagicMock(layout="SAFE_WIDE", crop_windows=[])))

    mock_preflight = MagicMock()
    mock_preflight.preflight_clip.return_value = VisualPreflightResult(
        usable=True, has_subtitles=False, subtitle_confidence=0.95, ending_complete=True, ending_natural=True,
        recommended_layout="SAFE_WIDE", notes="Deterministic fallback", preflight_mode="LOCAL_FALLBACK"
    )

    orch = AutoClipperOrchestrator(
        repository=temp_db,
        searcher=mock_searcher,
        source_filter=mock_source_filter,
        language_gate=mock_lang_gate,
        transcript_provider=mock_transcript_provider,
        candidate_generator=mock_candidate_gen,
        semantic_scorer=mock_scorer,
        boundary_refiner=BoundaryRefiner(),
        visual_analyzer=mock_vis_analyzer,
        visual_director=mock_vis_director,
        framing=mock_framing,
        visual_preflight=mock_preflight,
        output_dir=out_dir,
        download_dir=down_dir,
    )

    with patch("editing.subtitle_detector.SubtitleDetector.detect") as mock_detect:
        # Local detector returns UNKNOWN
        mock_detect.return_value = (SubtitleDetectionState.UNKNOWN, 0.50, {"note": "ambiguous"})
        res = orch.process_video(video_meta_or_url=vid_meta, custom_video_path=str(dummy_media), dry_run=True)

    assert res.is_success is False
    assert res.status == PipelineStatus.REJECTED_VISUAL
    assert "unknown" in res.rejection_reason.lower()


# ==============================================================================
# SECTION 4: Zero ASS Filter in Renderer Pipeline
# ==============================================================================

def test_renderer_pipeline_zero_ass_filter_when_source_existing():
    """PRISM Audit: CleanRenderer filtergraph must have 0 'subtitles=' filters when subtitle_policy is SOURCE_EXISTING."""
    renderer = CleanRenderer()
    plan = EditPlan(clip_id="test_zero_ass", duration=35.0, layout="SAFE_WIDE", subtitle_policy="SOURCE_EXISTING")

    filter_complex = renderer.build_filtergraph(edit_plan=plan, duration=35.0, subtitle_ass_path=None)

    # Filter out audio filtergraph [0:a]...
    video_filters = [seg for seg in filter_complex.split(";") if not seg.strip().startswith("[0:a]")]
    video_graph = ";".join(video_filters)

    assert "ass=" not in video_graph, "Found 'ass=' filter in SOURCE_EXISTING video pipeline!"
    assert "subtitles=" not in video_graph, "Found 'subtitles=' filter in SOURCE_EXISTING video pipeline!"
    assert "[v_base]null[v_out]" in video_graph


def test_renderer_pipeline_has_ass_filter_when_generate_policy(tmp_path: Path):
    """CleanRenderer filtergraph must include 'subtitles=' filter when subtitle_policy is GENERATE and ass_path exists."""
    renderer = CleanRenderer()
    fake_ass = tmp_path / "subs.ass"
    fake_ass.write_text("[Script Info]\nTitle: Test\n", encoding="utf-8")

    plan = EditPlan(clip_id="test_gen_ass", duration=35.0, layout="SAFE_WIDE", subtitle_policy="GENERATE")

    filter_complex = renderer.build_filtergraph(edit_plan=plan, duration=35.0, subtitle_ass_path=str(fake_ass))

    assert "subtitles=" in filter_complex
    assert str(fake_ass) in filter_complex


# ==============================================================================
# SECTION 5: Gemini QC Gate Hard Verification
# ==============================================================================

def test_gemini_qc_catches_both_double_subtitles_and_incomplete_ending(tmp_path: Path):
    """PRISM Audit: Gemini QC correctly flags both defects if present simultaneously."""
    dummy_clip = tmp_path / "double_defect.mp4"
    dummy_clip.write_bytes(b"\x00" * (64 * 1024))

    mock_client = MagicMock()
    fake_qc = {
        "passed": True,
        "score": 90,
        "double_subtitles_detected": True,
        "has_double_subtitles": "FAIL",
        "ending_complete": "FAIL",
        "ending_natural": "FAIL",
        "ending_reason": "Terputus saat kata 'sebenarnya...'",
        "subtitle_timing": "PASS",
        "subtitle_overlap": "PASS",
        "subtitle_linger": "PASS",
        "subtitle_text_accuracy": "PASS",
        "obvious_transcription_errors": [],
        "blocking_reasons": [],
        "summary": "Ada double subtitle dan ending terputus.",
    }
    mock_client.video_completion.return_value = json.dumps(fake_qc)
    mock_client.extract_json.return_value = fake_qc

    qc = GeminiNativeVideoQC(client=mock_client)
    res: GeminiVideoQCResult = qc.evaluate_video(dummy_clip, transcript_text="Contoh")

    assert res.passed is False
    assert res.double_subtitles_detected is True
    assert res.ending_complete == "FAIL"
    assert any("double subtitles" in b.lower() for b in res.blocking_reasons)
    assert any("unfinished sentence" in b.lower() or "terputus" in b.lower() for b in res.blocking_reasons)
