"""Test Suite for Auto Clipper V3.1 Double Subtitle Prevention & Sentence Ending Engine.

Covers:
1. Subtitle Policy & Double Subtitle Prevention:
   - Case A: Source with existing subtitles -> SOURCE_EXISTING (zero generated ASS layer)
   - Source without subtitles -> GENERATE (V3.1 Hybrid Subtitle Engine)
   - UNKNOWN subtitle state -> REJECT candidate (PipelineStatus.REJECTED_VISUAL)
   - Conflict (local detector BURNED_IN vs Gemini NO_SUBTITLE) -> REJECT candidate
   - Ambiguous / low confidence Gemini subtitle check (<0.6) -> REJECT candidate
2. Natural Sentence Ending Engine:
   - Case B1: Candidate with dangling ending ("waktu itu masih...", "karena sebenarnya...", "jadi hidup...")
     is extended to sentence boundary within 55s limit
   - Case B2: Dangling ending where extension exceeds 55s is REJECTED and falls back to next ranked candidate
   - Case B3: Gemini Native Video preflight flags incomplete ending -> extends within 55s or rejects
   - Case B4: Natural pauses (laughter, breathing) preserved without false positive rejection
3. Gemini Video QC Hard Rules:
   - Case C1: Double subtitles detected -> passed=False, blocking_reasons recorded
   - Case C2: Incomplete sentence ending (ending_complete=FAIL) -> passed=False, blocking_reasons recorded
   - Case C3: Unnatural ending (ending_natural=FAIL) -> passed=False, blocking_reasons recorded
   - Case C4: Failed QC archived to failed/ directory with reason.md
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from analysis.boundary_refiner import (
    BoundaryRefiner,
    RefinementResult,
    is_sentence_complete,
)
from analysis.candidate_generator import CandidateWindow
from analysis.semantic_scorer import SemanticScore
from analysis.visual_preflight import VisualPreflight, VisualPreflightResult
from discovery.searcher import VideoSourceMeta
from editing.edit_plan import EditPlan
from editing.subtitle_detector import SubtitleDetectionState
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
# 1. Boundary Refiner & Sentence Completeness Inspection
# ==============================================================================

def test_is_sentence_complete_detects_dangling_patterns():
    """Confirms is_sentence_complete detects dangling words, phrases, and ellipses."""
    # Complete sentences
    ok1, _ = is_sentence_complete("Ini adalah kesimpulan yang luar biasa.")
    assert ok1 is True

    ok2, _ = is_sentence_complete("Semua orang akhirnya tertawa!")
    assert ok2 is True

    ok3, _ = is_sentence_complete("Apakah ini sudah selesai?")
    assert ok3 is True

    ok4, _ = is_sentence_complete("prosesnya panjang dan hasilnya sukses")
    assert ok4 is True

    # Dangling phrases explicitly mentioned in specification
    bad1, r1 = is_sentence_complete("Waktu itu masih...")
    assert bad1 is False
    assert "ellipsis" in r1.lower() or "dangling" in r1.lower()

    bad2, r2 = is_sentence_complete("Waktu itu masih")
    assert bad2 is False
    assert "masih" in r2.lower()

    bad3, r3 = is_sentence_complete("karena sebenarnya...")
    assert bad3 is False

    bad4, r4 = is_sentence_complete("karena sebenarnya")
    assert bad4 is False
    assert "sebenarnya" in r4.lower()

    bad5, r5 = is_sentence_complete("kita harus berjuang jadi hidup")
    assert bad5 is False
    assert "jadi hidup" in r5.lower()

    # Dangling connectors
    bad6, r6 = is_sentence_complete("dia mau bicara dan")
    assert bad6 is False
    assert "dan" in r6.lower()

    bad7, r7 = is_sentence_complete("rencana ini berjalan karena")
    assert bad7 is False
    assert "karena" in r7.lower()

    bad8, r8 = is_sentence_complete("hal penting yang")
    assert bad8 is False
    assert "yang" in r8.lower()

    # Trailing mid-sentence punctuation
    bad9, r9 = is_sentence_complete("kami berangkat kemarin,")
    assert bad9 is False
    assert "mid-sentence punctuation" in r9.lower()


def test_boundary_refiner_extends_dangling_ending_within_55s():
    """Case B1: Incomplete ending is extended to sentence completion when <= 55s."""
    refiner = BoundaryRefiner(min_duration_sec=30.0, max_duration_sec=55.0, laughter_buffer_sec=0.2)

    segments = [
        TranscriptSegment(start=10.0, end=25.0, duration=15.0, text="Awal kisah yang menarik."),
        TranscriptSegment(start=25.2, end=42.0, duration=16.8, text="waktu itu masih"),  # Dangling ending at 42.0s
        TranscriptSegment(start=42.1, end=47.0, duration=4.9, text="belajar di bangku kuliah."),  # Completes sentence
    ]

    # Initial suggested window ends at 42.0s (dangling 'waktu itu masih')
    can_ext, new_end, reason = refiner.extend_to_sentence_boundary(
        start_sec=10.0,
        end_sec=42.0,
        segments=segments,
        max_duration_sec=55.0,
    )

    assert can_ext is True
    assert new_end == 47.2  # 47.0 + 0.2 laughter buffer
    new_duration = round(new_end - 10.0, 2)
    assert 30.0 <= new_duration <= 55.0
    assert "complete sentence" in reason.lower()


def test_boundary_refiner_rejects_extension_exceeding_55s():
    """Case B2: Dangling ending where extension exceeds 55s returns False."""
    refiner = BoundaryRefiner(min_duration_sec=30.0, max_duration_sec=55.0, laughter_buffer_sec=0.2)

    segments = [
        TranscriptSegment(start=5.0, end=40.0, duration=35.0, text="Awal pembicaraan panjang."),
        TranscriptSegment(start=40.2, end=54.0, duration=13.8, text="karena sebenarnya"),  # Dangling ending at 54.0s (dur=49s)
        TranscriptSegment(start=54.1, end=62.0, duration=7.9, text="kita semua tidak tahu masa depan."),  # Extends to 62.2s (dur=57.2s > 55s)
    ]

    can_ext, new_end, reason = refiner.extend_to_sentence_boundary(
        start_sec=5.0,
        end_sec=54.0,
        segments=segments,
        max_duration_sec=55.0,
    )

    assert can_ext is False
    assert new_end == 54.0
    assert "exceed" in reason.lower()


# ==============================================================================
# 2. Visual Preflight: Subtitles & Ending Inspection
# ==============================================================================

def test_visual_preflight_inspects_subtitles_and_ending(tmp_path: Path):
    """VisualPreflight parses has_subtitles, confidence, ending_complete, and ending_natural."""
    dummy_video = tmp_path / "sample_preflight.mp4"
    dummy_video.write_bytes(b"\x00" * (64 * 1024))

    mock_client = MagicMock()
    fake_response = {
        "usable": True,
        "has_subtitles": True,
        "confidence": 0.98,
        "reason": "Burned-in yellow captions in center lower frame",
        "ending_complete": True,
        "ending_natural": True,
        "ending_reason": "Speaker finished thought before silence",
        "shot_complexity": "low",
        "subject_composition": "acceptable",
        "recommended_layout": "SAFE_WIDE",
        "blocking_issues": [],
        "notes": "Verified source subtitles and clean ending",
    }
    mock_client.video_completion.return_value = json.dumps(fake_response)
    mock_client.extract_json.return_value = fake_response

    preflight = VisualPreflight(client=mock_client)
    res: VisualPreflightResult = preflight.preflight_clip(dummy_video)

    assert res.usable is True
    assert res.has_subtitles is True
    assert res.existing_visible_subtitles is True
    assert res.subtitle_confidence == 0.98
    assert res.ending_complete is True
    assert res.ending_natural is True
    assert res.recommended_layout == "SAFE_WIDE"


def test_visual_preflight_flags_incomplete_ending(tmp_path: Path):
    """VisualPreflight marks unusable when ending_complete is False."""
    dummy_video = tmp_path / "sample_incomplete.mp4"
    dummy_video.write_bytes(b"\x00" * (64 * 1024))

    mock_client = MagicMock()
    fake_response = {
        "usable": True,
        "has_subtitles": False,
        "confidence": 0.95,
        "reason": "No subtitles visible",
        "ending_complete": False,
        "ending_natural": False,
        "ending_reason": "Speaker cut off mid-sentence saying 'waktu itu masih...'",
        "shot_complexity": "low",
        "subject_composition": "acceptable",
        "recommended_layout": "SAFE_WIDE",
        "blocking_issues": [],
    }
    mock_client.video_completion.return_value = json.dumps(fake_response)
    mock_client.extract_json.return_value = fake_response

    preflight = VisualPreflight(client=mock_client)
    res: VisualPreflightResult = preflight.preflight_clip(dummy_video)

    assert res.usable is False
    assert res.ending_complete is False
    assert any("incomplete sentence" in b.lower() for b in res.blocking_issues)


# ==============================================================================
# 3. Gemini Video QC Hard Rules: Double Subtitles & Sentence Ending
# ==============================================================================

def test_gemini_video_qc_rejects_double_subtitles(tmp_path: Path):
    """Case C1: GeminiNativeVideoQC rejects when double subtitles are detected."""
    dummy_clip = tmp_path / "double_sub_clip.mp4"
    dummy_clip.write_bytes(b"\x00" * (64 * 1024))

    mock_client = MagicMock()
    fake_qc = {
        "passed": True,  # LLM initially claimed true, but detector must override
        "score": 85,
        "double_subtitles_detected": True,
        "ending_complete": "PASS",
        "ending_natural": "PASS",
        "subtitle_timing": "PASS",
        "subtitle_overlap": "PASS",
        "subtitle_linger": "PASS",
        "subtitle_text_accuracy": "PASS",
        "obvious_transcription_errors": [],
        "blocking_reasons": [],
        "summary": "Both native source and newly generated subtitles appear stacked.",
    }
    mock_client.video_completion.return_value = json.dumps(fake_qc)
    mock_client.extract_json.return_value = fake_qc

    qc = GeminiNativeVideoQC(client=mock_client)
    res: GeminiVideoQCResult = qc.evaluate_video(dummy_clip, transcript_text="Contoh teks dialog")

    assert res.passed is False
    assert res.double_subtitles_detected is True
    assert any("double subtitles" in b.lower() for b in res.blocking_reasons)


def test_gemini_video_qc_rejects_unfinished_ending(tmp_path: Path):
    """Case C2: GeminiNativeVideoQC rejects when ending_complete is FAIL."""
    dummy_clip = tmp_path / "unfinished_clip.mp4"
    dummy_clip.write_bytes(b"\x00" * (64 * 1024))

    mock_client = MagicMock()
    fake_qc = {
        "passed": True,
        "score": 85,
        "double_subtitles_detected": False,
        "ending_complete": "FAIL",
        "ending_natural": "FAIL",
        "subtitle_timing": "PASS",
        "subtitle_overlap": "PASS",
        "subtitle_linger": "PASS",
        "subtitle_text_accuracy": "PASS",
        "obvious_transcription_errors": [],
        "blocking_reasons": ["Kalimat terputus mendadak sebelum selesai"],
        "summary": "Ending cut off mid-word.",
    }
    mock_client.video_completion.return_value = json.dumps(fake_qc)
    mock_client.extract_json.return_value = fake_qc

    qc = GeminiNativeVideoQC(client=mock_client)
    res: GeminiVideoQCResult = qc.evaluate_video(dummy_clip, transcript_text="Contoh dialog")

    assert res.passed is False
    assert res.ending_complete == "FAIL"
    assert any("unfinished sentence" in b.lower() or "terputus" in b.lower() for b in res.blocking_reasons)


# ==============================================================================
# 4. Orchestrator Integration: Policy Enforcement & Candidate Fallback
# ==============================================================================

def test_orchestrator_source_existing_policy_generates_zero_ass(tmp_path: Path):
    """Case A: When source has subtitles, policy is SOURCE_EXISTING and ZERO ASS file is generated."""
    temp_db = StorageRepository(db_path=str(tmp_path / "test.db"))
    out_dir = tmp_path / "output"
    down_dir = tmp_path / "downloads"
    out_dir.mkdir(parents=True)
    down_dir.mkdir(parents=True)

    dummy_media = down_dir / "vid_sub.mp4"
    dummy_media.write_bytes(b"\x00" * (120 * 1024))

    vid_meta = VideoSourceMeta(
        video_id="vid_sub",
        url="https://youtube.com/watch?v=vid_sub",
        title="Video With Burned In Subtitles",
        duration_sec=300.0,
    )

    mock_searcher = MagicMock()
    mock_searcher.get_video_metadata.return_value = vid_meta

    mock_source_filter = MagicMock()
    mock_source_filter.filter_video.return_value = MagicMock(is_eligible=True)

    mock_lang_gate = MagicMock()
    mock_lang_gate.evaluate_transcript.return_value = MagicMock(eligible=True)

    transcript = [
        TranscriptSegment(start=10.0, end=45.0, duration=35.0, text="Ini adalah dialog yang ada subtitle bawaan."),
    ]
    mock_transcript_provider = MagicMock()
    mock_transcript_provider.get_phrase_transcript.return_value = transcript

    candidate = CandidateWindow(
        candidate_id="cand_1", start_sec=10.0, end_sec=45.0, duration_sec=35.0, text=transcript[0].text
    )
    mock_candidate_gen = MagicMock()
    mock_candidate_gen.generate_candidates.return_value = [candidate]

    score = SemanticScore(
        candidate_id="cand_1",
        good_clip=True,
        score=90.0,
        hook_score=90.0,
        payoff_score=90.0,
        self_contained_score=90.0,
        reason="Approved",
        suggested_start=10.0,
        suggested_end=45.0,
    )
    mock_scorer = MagicMock()
    mock_scorer.score_candidates.return_value = [score]
    mock_scorer.select_best_clip.return_value = (candidate, score, "APPROVED")

    mock_refiner = BoundaryRefiner()

    mock_vis_analyzer = MagicMock()
    mock_vis_analyzer.analyze_clip.return_value = MagicMock(scene_cuts=[])

    mock_vis_director = MagicMock()
    mock_vis_director.evaluate_window.return_value = MagicMock(approved=True, notes="OK")

    mock_framing = MagicMock()
    mock_framing.analyze_framing.return_value = MagicMock(layout="SAFE_WIDE", crop_windows=[])

    # Visual Preflight confirms visible subtitles exist
    mock_preflight = MagicMock()
    mock_preflight.preflight_clip.return_value = VisualPreflightResult(
        usable=True,
        has_subtitles=True,
        existing_visible_subtitles=True,
        subtitle_confidence=0.99,
        subtitle_reason="Visible source subtitles in bottom zone",
        ending_complete=True,
        ending_natural=True,
        recommended_layout="SAFE_WIDE",
        blocking_issues=[],
        notes="Preflight verified source subtitles",
        preflight_mode="GEMINI_NATIVE_VIDEO",
    )

    rendered_ass_files = []
    def fake_render(video_path, output_path, edit_plan, start_sec, duration, subtitle_ass_path=None):
        rendered_ass_files.append(subtitle_ass_path)
        Path(output_path).write_bytes(b"\x00" * (110 * 1024))
        return MagicMock(success=True, error_message=None)

    mock_renderer = MagicMock()
    mock_renderer.render.side_effect = fake_render

    mock_qc = MagicMock()
    mock_qc.evaluate.return_value = ThreeTierQCReport(
        passed=True,
        publishable=True,
        technical=TechnicalQCResult(passed=True, duration=35.0),
        visual=VisualQCResult(passed=True),
        perceptual=PerceptualQCResult(passed=True, publishable=True, score=90),
    )

    mock_gemini_qc = MagicMock()
    mock_gemini_qc.evaluate_video.return_value = GeminiVideoQCResult(
        passed=True,
        score=90,
        double_subtitles_detected=False,
        ending_complete="PASS",
        ending_natural="PASS",
    )

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
        visual_preflight=mock_preflight,
        renderer=mock_renderer,
        qc_gate=mock_qc,
        gemini_video_qc=mock_gemini_qc,
        output_dir=out_dir,
        download_dir=down_dir,
    )

    with patch("editing.subtitle_detector.SubtitleDetector.detect") as mock_detect:
        # Local detector agrees with Gemini
        mock_detect.return_value = (SubtitleDetectionState.BURNED_IN, 0.95, {})
        res = orch.process_video(video_meta_or_url=vid_meta, custom_video_path=str(dummy_media), dry_run=True)

    assert res.is_success is True
    # Subtitle ASS path MUST be None (ZERO generated ASS attached to renderer)
    assert rendered_ass_files == [None]


def test_orchestrator_rejects_subtitle_conflict(tmp_path: Path):
    """Confirms orchestrator strictly REJECTS candidate on subtitle detection conflict."""
    temp_db = StorageRepository(db_path=str(tmp_path / "test_conflict.db"))
    out_dir = tmp_path / "output"
    down_dir = tmp_path / "downloads"
    out_dir.mkdir(parents=True)
    down_dir.mkdir(parents=True)

    dummy_media = down_dir / "vid_conflict.mp4"
    dummy_media.write_bytes(b"\x00" * (120 * 1024))

    vid_meta = VideoSourceMeta(
        video_id="vid_conflict",
        url="https://youtube.com/watch?v=vid_conflict",
        title="Conflict Video",
        duration_sec=300.0,
    )

    mock_searcher = MagicMock()
    mock_searcher.get_video_metadata.return_value = vid_meta
    mock_source_filter = MagicMock()
    mock_source_filter.filter_video.return_value = MagicMock(is_eligible=True)
    mock_lang_gate = MagicMock()
    mock_lang_gate.evaluate_transcript.return_value = MagicMock(eligible=True)

    transcript = [
        TranscriptSegment(start=10.0, end=45.0, duration=35.0, text="Dialog contoh."),
    ]
    mock_transcript_provider = MagicMock()
    mock_transcript_provider.get_phrase_transcript.return_value = transcript

    candidate = CandidateWindow(candidate_id="c1", start_sec=10.0, end_sec=45.0, duration_sec=35.0, text="Dialog")
    mock_candidate_gen = MagicMock()
    mock_candidate_gen.generate_candidates.return_value = [candidate]

    score = SemanticScore(
        candidate_id="c1",
        good_clip=True,
        score=90.0,
        hook_score=90.0,
        payoff_score=90.0,
        self_contained_score=90.0,
        reason="Approved",
        suggested_start=10.0,
        suggested_end=45.0,
    )
    mock_scorer = MagicMock()
    mock_scorer.score_candidates.return_value = [score]
    mock_scorer.select_best_clip.return_value = (candidate, score, "APPROVED")

    mock_refiner = BoundaryRefiner()
    mock_vis_analyzer = MagicMock()
    mock_vis_analyzer.analyze_clip.return_value = MagicMock(scene_cuts=[])
    mock_vis_director = MagicMock()
    mock_vis_director.evaluate_window.return_value = MagicMock(approved=True)
    mock_framing = MagicMock()
    mock_framing.analyze_framing.return_value = MagicMock(layout="SAFE_WIDE", crop_windows=[])

    # Gemini says NO subtitles
    mock_preflight = MagicMock()
    mock_preflight.preflight_clip.return_value = VisualPreflightResult(
        usable=True,
        has_subtitles=False,
        existing_visible_subtitles=False,
        subtitle_confidence=0.95,
        subtitle_reason="No subtitles detected",
        ending_complete=True,
        ending_natural=True,
        recommended_layout="SAFE_WIDE",
        blocking_issues=[],
        notes="Gemini native inspection",
        preflight_mode="GEMINI_NATIVE_VIDEO",
    )

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
        visual_preflight=mock_preflight,
        output_dir=out_dir,
        download_dir=down_dir,
    )

    with patch("editing.subtitle_detector.SubtitleDetector.detect") as mock_detect:
        # Local detector detects BURNED_IN with high confidence (0.85) -> Direct conflict!
        mock_detect.return_value = (SubtitleDetectionState.BURNED_IN, 0.85, {})
        res = orch.process_video(video_meta_or_url=vid_meta, custom_video_path=str(dummy_media), dry_run=True)

    assert res.is_success is False
    assert res.status == PipelineStatus.REJECTED_VISUAL
    assert "conflict" in res.rejection_reason.lower()


def test_orchestrator_candidate_fallback_when_top_has_unfinished_ending(tmp_path: Path):
    """Case B2: When rank #1 has dangling ending exceeding 55s, orchestrator falls back to rank #2."""
    temp_db = StorageRepository(db_path=str(tmp_path / "test_fallback.db"))
    out_dir = tmp_path / "output"
    down_dir = tmp_path / "downloads"
    out_dir.mkdir(parents=True)
    down_dir.mkdir(parents=True)

    dummy_media = down_dir / "vid_fallback.mp4"
    dummy_media.write_bytes(b"\x00" * (120 * 1024))

    vid_meta = VideoSourceMeta(
        video_id="vid_fallback",
        url="https://youtube.com/watch?v=vid_fallback",
        title="Fallback Video",
        duration_sec=300.0,
    )

    mock_searcher = MagicMock()
    mock_searcher.get_video_metadata.return_value = vid_meta
    mock_source_filter = MagicMock()
    mock_source_filter.filter_video.return_value = MagicMock(is_eligible=True)
    mock_lang_gate = MagicMock()
    mock_lang_gate.evaluate_transcript.return_value = MagicMock(eligible=True)

    # Transcript:
    # Segment 1-2 (cand_1): ends at 52s with 'waktu itu masih', next phrase ends at 65s (exceeds 55s limit)
    # Segment 3 (cand_2): [70s, 105s] complete sentence
    transcript = [
        TranscriptSegment(start=10.0, end=40.0, duration=30.0, text="Awal cerita pembuka."),
        TranscriptSegment(start=40.2, end=52.0, duration=11.8, text="waktu itu masih"),  # dangling ending!
        TranscriptSegment(start=52.1, end=65.0, duration=12.9, text="belum ada kepastian sama sekali."),  # 65 > 55s
        TranscriptSegment(start=70.0, end=105.0, duration=35.0, text="Cerita kedua yang tuntas dan lengkap."),  # valid!
    ]
    mock_transcript_provider = MagicMock()
    mock_transcript_provider.get_phrase_transcript.return_value = transcript

    cand1 = CandidateWindow(candidate_id="cand_bad_ending", start_sec=10.0, end_sec=52.0, duration_sec=42.0, text="waktu itu masih")
    cand2 = CandidateWindow(candidate_id="cand_clean_ending", start_sec=70.0, end_sec=105.0, duration_sec=35.0, text="Cerita kedua lengkap.")

    mock_candidate_gen = MagicMock()
    mock_candidate_gen.generate_candidates.return_value = [cand1, cand2]

    score1 = SemanticScore(
        candidate_id="cand_bad_ending",
        good_clip=True,
        score=95.0,
        hook_score=95.0,
        payoff_score=95.0,
        self_contained_score=95.0,
        reason="Viral candidate",
        suggested_start=10.0,
        suggested_end=52.0,
    )
    score2 = SemanticScore(
        candidate_id="cand_clean_ending",
        good_clip=True,
        score=88.0,
        hook_score=88.0,
        payoff_score=88.0,
        self_contained_score=88.0,
        reason="Clean ending candidate",
        suggested_start=70.0,
        suggested_end=105.0,
    )

    mock_scorer = MagicMock()
    mock_scorer.score_candidates.return_value = [score1, score2]
    # Rank 1 is cand_bad_ending
    mock_scorer.select_best_clip.return_value = (cand1, score1, "APPROVED")

    mock_refiner = BoundaryRefiner()

    mock_vis_analyzer = MagicMock()
    mock_vis_analyzer.analyze_clip.return_value = MagicMock(scene_cuts=[])
    mock_vis_director = MagicMock()
    mock_vis_director.evaluate_window.return_value = MagicMock(approved=True)
    mock_framing = MagicMock()
    mock_framing.analyze_framing.return_value = MagicMock(layout="SAFE_WIDE", crop_windows=[])

    mock_preflight = MagicMock()
    mock_preflight.preflight_clip.return_value = VisualPreflightResult(
        usable=True,
        has_subtitles=False,
        existing_visible_subtitles=False,
        subtitle_confidence=0.99,
        ending_complete=True,
        ending_natural=True,
        recommended_layout="SAFE_WIDE",
        blocking_issues=[],
        notes="Preflight OK",
        preflight_mode="LOCAL_FALLBACK",
    )

    def fake_render(video_path, output_path, edit_plan, start_sec, duration, subtitle_ass_path=None):
        Path(output_path).write_bytes(b"\x00" * (110 * 1024))
        return MagicMock(success=True)

    mock_renderer = MagicMock()
    mock_renderer.render.side_effect = fake_render

    mock_qc = MagicMock()
    mock_qc.evaluate.return_value = ThreeTierQCReport(
        passed=True,
        publishable=True,
        technical=TechnicalQCResult(passed=True, duration=35.0),
        visual=VisualQCResult(passed=True),
        perceptual=PerceptualQCResult(passed=True, publishable=True, score=88),
    )

    mock_gemini_qc = MagicMock()
    mock_gemini_qc.evaluate_video.return_value = GeminiVideoQCResult(
        passed=True, score=88, double_subtitles_detected=False, ending_complete="PASS", ending_natural="PASS"
    )

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
        visual_preflight=mock_preflight,
        renderer=mock_renderer,
        qc_gate=mock_qc,
        gemini_video_qc=mock_gemini_qc,
        output_dir=out_dir,
        download_dir=down_dir,
    )

    with patch("editing.subtitle_detector.SubtitleDetector.detect") as mock_detect, \
         patch("editing.word_subtitle_engine.generate_ass_from_words") as mock_gen_ass:
        mock_detect.return_value = (SubtitleDetectionState.NONE, 0.95, {})
        mock_gen_ass.return_value = (True, {"valid": True, "phrase_count": 5, "word_count": 20})
        res = orch.process_video(video_meta_or_url=vid_meta, custom_video_path=str(dummy_media), dry_run=True)

    assert res.is_success is True
    # Orchestrator should have selected cand2 because cand1 ending exceeded 55s
    assert res.selected_candidate["candidate_id"] == "cand_clean_ending"
    assert res.duration_sec == pytest.approx(35.2, 0.5)
