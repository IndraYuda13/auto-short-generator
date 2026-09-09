"""PRISM Independent Quality Assurance & Adversarial Verification Suite for Auto Clipper V3.1.

Verification Target:
Auto Clipper V3.1 Subtitle Policy & Double Subtitle Prevention + Natural Sentence Ending Engine.

Test Domains:
1. Subtitle Policy & Double Subtitle Prevention:
   - Source clips with embedded subtitles -> SOURCE_EXISTING (0 ASS)
   - Source clips with burned-in subtitles (Gemini verified) -> SOURCE_EXISTING (0 ASS)
   - Source clips without subtitles -> GENERATE (V3.1 Hybrid Subtitle Engine)
   - Local UNKNOWN subtitle state -> REJECT candidate (PipelineStatus.REJECTED_VISUAL)
   - Gemini low-confidence (<0.7) subtitle detection -> REJECT candidate (PipelineStatus.REJECTED_VISUAL)
   - Local BURNED_IN vs Gemini NO_SUBTITLES conflict -> REJECT candidate (PipelineStatus.REJECTED_VISUAL)
   - Local NONE vs Gemini HAS_SUBTITLES conflict -> REJECT candidate (PipelineStatus.REJECTED_VISUAL)
   - Gemini Video QC catches double subtitles (double_subtitles_detected=True / has_double_subtitles=FAIL)
   - Orchestrator archives double-subtitle QC failures to failed/ directory and prevents upload

2. Natural Sentence Ending Engine:
   - Exhaustive Indonesian dangling word detection (conjunctions, prepositions, auxiliary verbs, degree words)
   - Indonesian dangling phrase detection ("waktu itu masih", "karena sebenarnya", "jadi hidup", etc.)
   - Trailing punctuation check (ellipsis, comma, semicolon, colon, dash)
   - Non-dangling complete sentences with terminal punctuation or complete semantic thoughts
   - Natural speech pauses & laughter preservation (no false-positive rejection on haha/laughter)
   - BoundaryRefiner extension within 55s limit (accumulates segments until complete)
   - BoundaryRefiner rejection when extension exceeds 55s limit
   - BoundaryRefiner rejection when transcript ends mid-sentence
   - Orchestrator candidate fallback: Rank #1 dangling (>55s) falls back to Rank #2
   - Orchestrator candidate exhaustion: All candidates dangling (>55s) -> REJECTED_BOUNDARY
   - Gemini Visual Preflight flags incomplete ending -> attempts extension or REJECTED_VISUAL
   - Gemini Video QC flags unfinished ending (ending_complete=FAIL / ending_natural=FAIL) -> QC_FAILED

3. Regression Matrix & End-to-End Pipeline Integrity:
   - Full matrix verification across Cases A, B, and C
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from analysis.boundary_refiner import (
    BoundaryRefiner,
    RefinementResult,
    is_sentence_complete,
    DANGLING_CONNECTORS,
    DANGLING_PHRASES,
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
# DOMAIN 1: Subtitle Policy & Double Subtitle Prevention Verification
# ==============================================================================

class TestSubtitlePolicyVerification:
    """Verifies subtitle policy classification, zero ASS generation, and conflict handling."""

    @pytest.fixture
    def test_env(self, tmp_path: Path):
        db_file = tmp_path / "prism_test.db"
        repo = StorageRepository(db_path=str(db_file))
        out_dir = tmp_path / "output"
        down_dir = tmp_path / "downloads"
        out_dir.mkdir(parents=True, exist_ok=True)
        down_dir.mkdir(parents=True, exist_ok=True)

        media = down_dir / "test_video.mp4"
        media.write_bytes(b"\x00" * (100 * 1024))

        meta = VideoSourceMeta(
            video_id="vid_sub_prism",
            url="https://youtube.com/watch?v=vid_sub_prism",
            title="Video Test Subtitle Policy",
            duration_sec=300.0,
        )

        return {
            "repo": repo,
            "out_dir": out_dir,
            "down_dir": down_dir,
            "media": media,
            "meta": meta,
        }

    def _setup_base_orchestrator(self, test_env, preflight_result, segments=None):
        meta = test_env["meta"]
        media = test_env["media"]

        mock_searcher = MagicMock()
        mock_searcher.get_video_metadata.return_value = meta

        mock_filter = MagicMock()
        mock_filter.filter_video.return_value = MagicMock(is_eligible=True)

        mock_lang = MagicMock()
        mock_lang.evaluate_transcript.return_value = MagicMock(eligible=True)

        if segments is None:
            segments = [
                TranscriptSegment(start=10.0, end=45.0, duration=35.0, text="Ini adalah kalimat pengujian yang tuntas."),
            ]
        mock_tp = MagicMock()
        mock_tp.get_phrase_transcript.return_value = segments

        cand = CandidateWindow(candidate_id="c_sub", start_sec=10.0, end_sec=45.0, duration_sec=35.0, text=segments[0].text)
        mock_cg = MagicMock()
        mock_cg.generate_candidates.return_value = [cand]

        score = SemanticScore(
            candidate_id="c_sub",
            good_clip=True,
            score=92.0,
            hook_score=92.0,
            payoff_score=92.0,
            self_contained_score=92.0,
            reason="Approved",
            suggested_start=10.0,
            suggested_end=45.0,
        )
        mock_scorer = MagicMock()
        mock_scorer.score_candidates.return_value = [score]
        mock_scorer.select_best_clip.return_value = (cand, score, "APPROVED")

        mock_refiner = BoundaryRefiner(min_duration_sec=30.0, max_duration_sec=55.0)

        mock_va = MagicMock()
        mock_va.analyze_clip.return_value = MagicMock(scene_cuts=[])

        mock_vd = MagicMock()
        mock_vd.evaluate_window.return_value = MagicMock(approved=True, notes="OK")

        mock_framing = MagicMock()
        mock_framing.analyze_framing.return_value = MagicMock(layout="SAFE_WIDE", crop_windows=[])

        mock_preflight = MagicMock()
        mock_preflight.preflight_clip.return_value = preflight_result

        rendered_sub_paths = []
        def fake_render(video_path, output_path, edit_plan, start_sec, duration, subtitle_ass_path=None):
            rendered_sub_paths.append(subtitle_ass_path)
            Path(output_path).write_bytes(b"\x00" * (120 * 1024))
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
            repository=test_env["repo"],
            searcher=mock_searcher,
            source_filter=mock_filter,
            language_gate=mock_lang,
            transcript_provider=mock_tp,
            candidate_generator=mock_cg,
            semantic_scorer=mock_scorer,
            boundary_refiner=mock_refiner,
            visual_analyzer=mock_va,
            visual_director=mock_vd,
            framing=mock_framing,
            visual_preflight=mock_preflight,
            renderer=mock_renderer,
            qc_gate=mock_qc,
            gemini_video_qc=mock_gemini_qc,
            output_dir=test_env["out_dir"],
            download_dir=test_env["down_dir"],
        )

        return orch, rendered_sub_paths

    def test_case_a_embedded_subtitle_forces_source_existing_zero_ass(self, test_env):
        """Case A1: Embedded subtitle stream -> SOURCE_EXISTING with 0 generated ASS."""
        preflight = VisualPreflightResult(
            usable=True,
            has_subtitles=False,  # Even if Gemini saw nothing, embedded track wins
            existing_visible_subtitles=False,
            subtitle_confidence=0.9,
            ending_complete=True,
            ending_natural=True,
            recommended_layout="SAFE_WIDE",
            notes="Gemini preflight",
            preflight_mode="GEMINI_NATIVE_VIDEO",
        )
        orch, rendered_subs = self._setup_base_orchestrator(test_env, preflight)

        with patch("editing.subtitle_detector.SubtitleDetector.detect") as mock_detect:
            mock_detect.return_value = (SubtitleDetectionState.EMBEDDED_TRACK, 1.0, {"format": "mov_text"})
            res = orch.process_video(video_meta_or_url=test_env["meta"], custom_video_path=str(test_env["media"]), dry_run=True)

        assert res.is_success is True
        assert res.status == PipelineStatus.COMPLETED
        # Verify ZERO ASS file passed to renderer
        assert rendered_subs == [None]

    def test_case_a_burned_in_subtitles_forces_source_existing_zero_ass(self, test_env):
        """Case A2: Source clip with burned-in subtitles -> SOURCE_EXISTING with 0 generated ASS."""
        preflight = VisualPreflightResult(
            usable=True,
            has_subtitles=True,
            existing_visible_subtitles=True,
            subtitle_confidence=0.97,
            subtitle_reason="Visible bottom burned-in captions in source video",
            ending_complete=True,
            ending_natural=True,
            recommended_layout="SAFE_WIDE",
            notes="Gemini preflight verified subtitles",
            preflight_mode="GEMINI_NATIVE_VIDEO",
        )
        orch, rendered_subs = self._setup_base_orchestrator(test_env, preflight)

        with patch("editing.subtitle_detector.SubtitleDetector.detect") as mock_detect:
            mock_detect.return_value = (SubtitleDetectionState.BURNED_IN, 0.95, {})
            res = orch.process_video(video_meta_or_url=test_env["meta"], custom_video_path=str(test_env["media"]), dry_run=True)

        assert res.is_success is True
        assert rendered_subs == [None]

    def test_source_without_subtitles_triggers_generate_policy(self, test_env):
        """Source clip without subtitles -> GENERATE with Hybrid Subtitle Engine."""
        preflight = VisualPreflightResult(
            usable=True,
            has_subtitles=False,
            existing_visible_subtitles=False,
            subtitle_confidence=0.95,
            subtitle_reason="No visible text detected",
            ending_complete=True,
            ending_natural=True,
            recommended_layout="SAFE_WIDE",
            notes="Gemini preflight clean",
            preflight_mode="GEMINI_NATIVE_VIDEO",
        )
        orch, rendered_subs = self._setup_base_orchestrator(test_env, preflight)

        with patch("editing.subtitle_detector.SubtitleDetector.detect") as mock_detect, \
             patch("editing.word_subtitle_engine.generate_ass_from_words") as mock_gen_ass:
            mock_detect.return_value = (SubtitleDetectionState.NONE, 0.95, {})
            mock_gen_ass.return_value = (True, {"valid": True, "phrase_count": 8, "word_count": 32})
            res = orch.process_video(video_meta_or_url=test_env["meta"], custom_video_path=str(test_env["media"]), dry_run=True)

        assert res.is_success is True
        assert len(rendered_subs) == 1
        assert rendered_subs[0] is not None
        assert rendered_subs[0].endswith(".ass")

    def test_unknown_subtitle_state_triggers_immediate_rejection(self, test_env):
        """UNKNOWN subtitle state triggers immediate candidate rejection (NEVER guess)."""
        preflight = VisualPreflightResult(
            usable=True,
            has_subtitles=False,
            existing_visible_subtitles=False,
            subtitle_confidence=0.9,
            ending_complete=True,
            ending_natural=True,
            recommended_layout="SAFE_WIDE",
            notes="Gemini preflight",
            preflight_mode="GEMINI_NATIVE_VIDEO",
        )
        orch, _ = self._setup_base_orchestrator(test_env, preflight)

        with patch("editing.subtitle_detector.SubtitleDetector.detect") as mock_detect:
            mock_detect.return_value = (SubtitleDetectionState.UNKNOWN, 0.5, {})
            res = orch.process_video(video_meta_or_url=test_env["meta"], custom_video_path=str(test_env["media"]), dry_run=True)

        assert res.is_success is False
        assert res.status == PipelineStatus.REJECTED_VISUAL
        assert "UNKNOWN" in (res.rejection_reason or "")

    def test_ambiguous_gemini_subtitle_confidence_triggers_rejection(self, test_env):
        """Gemini subtitle confidence < 0.7 triggers immediate candidate rejection."""
        preflight = VisualPreflightResult(
            usable=True,
            has_subtitles=False,
            existing_visible_subtitles=False,
            subtitle_confidence=0.55,  # Low / ambiguous confidence
            subtitle_reason="Unclear if bottom text is watermark or caption",
            ending_complete=True,
            ending_natural=True,
            recommended_layout="SAFE_WIDE",
            notes="Gemini preflight ambiguous",
            preflight_mode="GEMINI_NATIVE_VIDEO",
        )
        orch, _ = self._setup_base_orchestrator(test_env, preflight)

        with patch("editing.subtitle_detector.SubtitleDetector.detect") as mock_detect:
            mock_detect.return_value = (SubtitleDetectionState.NONE, 0.8, {})
            res = orch.process_video(video_meta_or_url=test_env["meta"], custom_video_path=str(test_env["media"]), dry_run=True)

        assert res.is_success is False
        assert res.status == PipelineStatus.REJECTED_VISUAL
        rej = (res.rejection_reason or "").lower()
        assert "ambiguous" in rej or "low confidence" in rej

    def test_conflict_local_burned_in_vs_gemini_none_triggers_rejection(self, test_env):
        """Local BURNED_IN (conf>=0.7) vs Gemini NO_SUBTITLES (conf>=0.7) -> Conflict rejection."""
        preflight = VisualPreflightResult(
            usable=True,
            has_subtitles=False,  # Gemini says NO
            existing_visible_subtitles=False,
            subtitle_confidence=0.92,
            subtitle_reason="Clean frame, no subtitles",
            ending_complete=True,
            ending_natural=True,
            recommended_layout="SAFE_WIDE",
            notes="Gemini preflight",
            preflight_mode="GEMINI_NATIVE_VIDEO",
        )
        orch, _ = self._setup_base_orchestrator(test_env, preflight)

        with patch("editing.subtitle_detector.SubtitleDetector.detect") as mock_detect:
            mock_detect.return_value = (SubtitleDetectionState.BURNED_IN, 0.88, {})  # Local says YES
            res = orch.process_video(video_meta_or_url=test_env["meta"], custom_video_path=str(test_env["media"]), dry_run=True)

        assert res.is_success is False
        assert res.status == PipelineStatus.REJECTED_VISUAL
        assert "conflict" in (res.rejection_reason or "").lower()

    def test_conflict_local_none_vs_gemini_has_subtitles_triggers_rejection(self, test_env):
        """Local NONE (conf>=0.7) vs Gemini HAS_SUBTITLES (conf>=0.7) -> Conflict rejection."""
        preflight = VisualPreflightResult(
            usable=True,
            has_subtitles=True,  # Gemini says YES
            existing_visible_subtitles=True,
            subtitle_confidence=0.91,
            subtitle_reason="Found white subtitles",
            ending_complete=True,
            ending_natural=True,
            recommended_layout="SAFE_WIDE",
            notes="Gemini preflight",
            preflight_mode="GEMINI_NATIVE_VIDEO",
        )
        orch, _ = self._setup_base_orchestrator(test_env, preflight)

        with patch("editing.subtitle_detector.SubtitleDetector.detect") as mock_detect:
            mock_detect.return_value = (SubtitleDetectionState.NONE, 0.90, {})  # Local says NO
            res = orch.process_video(video_meta_or_url=test_env["meta"], custom_video_path=str(test_env["media"]), dry_run=True)

        assert res.is_success is False
        assert res.status == PipelineStatus.REJECTED_VISUAL
        assert "conflict" in (res.rejection_reason or "").lower()


# ==============================================================================
# DOMAIN 2: Natural Sentence Ending Engine Verification
# ==============================================================================

class TestNaturalSentenceEndingEngine:
    """Verifies linguistic sentence ending detection, pause preservation, and boundary extension."""

    def test_exhaustive_dangling_connector_matrix(self):
        """Verifies all registered Indonesian connectors in DANGLING_CONNECTORS are flagged."""
        # Test every single connector in DANGLING_CONNECTORS
        for connector in DANGLING_CONNECTORS:
            text = f"Hari ini kita pergi ke sekolah {connector}"
            is_comp, reason = is_sentence_complete(text)
            assert is_comp is False, f"Expected connector '{connector}' to be flagged, got is_complete=True"
            assert connector in reason.lower()

    def test_exhaustive_dangling_phrase_matrix(self):
        """Verifies all registered phrases in DANGLING_PHRASES are flagged."""
        for phrase in DANGLING_PHRASES:
            text = f"Pengalaman hidup saya {phrase}"
            is_comp, reason = is_sentence_complete(text)
            assert is_comp is False, f"Expected phrase '{phrase}' to be flagged, got is_complete=True"
            assert "phrase" in reason.lower() or phrase in reason.lower()

    def test_dangling_with_trailing_punctuation(self):
        """Ellipses and mid-sentence punctuation are flagged even if word seems ok."""
        cases = [
            ("Kehidupan berjalan sangat indah...", "ellipsis"),
            ("Mereka pergi ke kota…", "ellipsis"),
            ("Pekerjaan ini belum beres,", "mid-sentence punctuation"),
            ("Ada banyak alternatif;", "mid-sentence punctuation"),
            ("Berikut adalah rinciannya:", "mid-sentence punctuation"),
            ("Kami menunggu kabar-", "mid-sentence punctuation"),
        ]
        for text, expected in cases:
            is_comp, reason = is_sentence_complete(text)
            assert is_comp is False, f"Failed for '{text}'"
            assert expected in reason.lower()

    def test_complete_sentences_with_terminal_punctuation(self):
        """Sentences ending in period, question mark, or exclamation mark pass."""
        good_cases = [
            "Ini adalah akhir dari pembahasan kita.",
            "Apakah ada pertanyaan lanjutan?",
            "Sungguh pencapaian yang mengagumkan!",
            "Proyek akhirnya selesai tepat waktu.",
        ]
        for text in good_cases:
            is_comp, _ = is_sentence_complete(text)
            assert is_comp is True, f"Expected True for '{text}'"

    def test_natural_speech_pauses_and_reactions_not_misidentified(self):
        """Natural laughter, breathing, reaction sounds do not trigger false positive rejection."""
        reaction_cases = [
            "Semua orang tertawa haha",
            "Momen itu sangat lucu hahaha",
            "Kita semua tidak bisa menahan tawa",
            "Dan penonton bersorak gembira",
        ]
        for text in reaction_cases:
            is_comp, _ = is_sentence_complete(text)
            assert is_comp is True, f"Expected True for '{text}'"

    def test_boundary_refiner_multi_segment_extension_within_55s(self):
        """BoundaryRefiner extends across multiple segments until sentence finishes within 55s."""
        refiner = BoundaryRefiner(min_duration_sec=30.0, max_duration_sec=55.0, laughter_buffer_sec=0.2)

        # 10s to 40s (dur=30s), ends with 'dan'
        # next seg: 40.2s to 45.0s, ends with 'masih' (still dangling!)
        # next seg: 45.1s to 50.0s, ends with 'harus berusaha.' (complete!)
        segments = [
            TranscriptSegment(start=10.0, end=40.0, duration=30.0, text="Kami memulai perjalanan panjang dan"),
            TranscriptSegment(start=40.2, end=45.0, duration=4.8, text="di perjalanan kami masih"),
            TranscriptSegment(start=45.1, end=50.0, duration=4.9, text="harus berusaha keras sampai tiba."),
            TranscriptSegment(start=50.5, end=60.0, duration=9.5, text="Bagian berikutnya sama sekali baru."),
        ]

        can_ext, new_end, reason = refiner.extend_to_sentence_boundary(
            start_sec=10.0,
            end_sec=40.0,
            segments=segments,
            max_duration_sec=55.0,
        )

        assert can_ext is True
        assert new_end == 50.2  # 50.0 + 0.2 laughter buffer
        assert (new_end - 10.0) <= 55.0
        assert "complete sentence" in reason.lower()

    def test_boundary_refiner_rejects_when_transcript_exhausted(self):
        """BoundaryRefiner rejects when transcript ends with incomplete sentence."""
        refiner = BoundaryRefiner(min_duration_sec=30.0, max_duration_sec=55.0)

        segments = [
            TranscriptSegment(start=10.0, end=35.0, duration=25.0, text="Awal dialog pembicara"),
            TranscriptSegment(start=35.1, end=45.0, duration=9.9, text="karena sebenarnya..."),
        ]

        can_ext, new_end, reason = refiner.extend_to_sentence_boundary(
            start_sec=10.0,
            end_sec=45.0,
            segments=segments,
            max_duration_sec=55.0,
        )

        assert can_ext is False
        assert new_end == 45.0
        assert "end of transcript reached" in reason.lower()

    def test_refine_method_automatically_triggers_extension(self):
        """BoundaryRefiner.refine() automatically performs extension on dangling ending."""
        refiner = BoundaryRefiner(min_duration_sec=30.0, max_duration_sec=55.0, laughter_buffer_sec=0.2)

        segments = [
            TranscriptSegment(start=10.0, end=38.0, duration=28.0, text="Penjelasan pertama sudah jelas."),
            TranscriptSegment(start=38.1, end=44.0, duration=5.9, text="waktu itu masih"),  # dangling at 44s
            TranscriptSegment(start=44.1, end=49.0, duration=4.9, text="sangat sedikit orang yang tahu."),  # completes at 49s
        ]

        res = refiner.refine(start_sec=10.0, end_sec=44.0, segments=segments)

        assert res.is_valid is True
        assert res.sentence_complete is True
        assert res.refined_end == 49.2  # 49.0 + 0.2 buffer
        assert res.duration == pytest.approx(39.2, 0.1)

    def test_refine_method_rejects_dangling_ending_when_extension_exceeds_55s(self):
        """BoundaryRefiner.refine() returns is_valid=False when extension exceeds 55s."""
        refiner = BoundaryRefiner(min_duration_sec=30.0, max_duration_sec=55.0, laughter_buffer_sec=0.2)

        segments = [
            TranscriptSegment(start=10.0, end=50.0, duration=40.0, text="Pembahasan mendalam sejak awal."),
            TranscriptSegment(start=50.1, end=54.0, duration=3.9, text="karena sebenarnya"),  # dur=44s, dangling
            TranscriptSegment(start=54.1, end=68.0, duration=13.9, text="hal tersebut membutuhkan penelitian berbulan-bulan."),  # ends at 68.2s (>55s)
        ]

        res = refiner.refine(start_sec=10.0, end_sec=54.0, segments=segments)

        assert res.is_valid is False
        assert res.sentence_complete is False
        assert "unfinished sentence" in (res.rejection_reason or "").lower()


# ==============================================================================
# DOMAIN 3: Gemini Video QC & Orchestrator Rejection Gate Verification
# ==============================================================================

class TestGeminiVideoQCGates:
    """Verifies Gemini Native Video QC hard rules and orchestrator failure handling."""

    def test_gemini_qc_catches_double_subtitles_hard_fail(self, tmp_path: Path):
        """Case C1: Gemini QC catches double subtitles and overrides LLM passed=true."""
        dummy_clip = tmp_path / "double_sub.mp4"
        dummy_clip.write_bytes(b"\x00" * (64 * 1024))

        mock_client = MagicMock()
        # LLM hallucinating passed=True despite detecting double subtitles
        fake_llm_json = {
            "passed": True,
            "score": 85,
            "double_subtitles_detected": True,
            "has_double_subtitles": "FAIL",
            "ending_complete": "PASS",
            "ending_natural": "PASS",
            "subtitle_timing": "PASS",
            "subtitle_overlap": "PASS",
            "subtitle_linger": "PASS",
            "subtitle_text_accuracy": "PASS",
            "obvious_transcription_errors": [],
            "blocking_reasons": [],
            "summary": "Source burned-in captions present beneath newly rendered ASS titles.",
        }
        mock_client.video_completion.return_value = json.dumps(fake_llm_json)
        mock_client.extract_json.return_value = fake_llm_json

        qc = GeminiNativeVideoQC(client=mock_client)
        res = qc.evaluate_video(dummy_clip, transcript_text="Contoh teks dialog")

        # Hard invariant: passed MUST be False
        assert res.passed is False
        assert res.double_subtitles_detected is True
        assert res.has_double_subtitles == "FAIL"
        assert any("double subtitles" in b.lower() for b in res.blocking_reasons)

    def test_gemini_qc_catches_incomplete_sentence_ending_hard_fail(self, tmp_path: Path):
        """Case C2: Gemini QC catches unfinished sentence ending and overrides passed=true."""
        dummy_clip = tmp_path / "incomplete_ending.mp4"
        dummy_clip.write_bytes(b"\x00" * (64 * 1024))

        mock_client = MagicMock()
        fake_llm_json = {
            "passed": True,
            "score": 82,
            "double_subtitles_detected": False,
            "has_double_subtitles": "PASS",
            "ending_complete": "FAIL",
            "ending_natural": "FAIL",
            "ending_reason": "Pembicara terputus saat berkata 'waktu itu masih...'",
            "subtitle_timing": "PASS",
            "subtitle_overlap": "PASS",
            "subtitle_linger": "PASS",
            "subtitle_text_accuracy": "PASS",
            "obvious_transcription_errors": [],
            "blocking_reasons": [],
        }
        mock_client.video_completion.return_value = json.dumps(fake_llm_json)
        mock_client.extract_json.return_value = fake_llm_json

        qc = GeminiNativeVideoQC(client=mock_client)
        res = qc.evaluate_video(dummy_clip, transcript_text="Dialog contoh")

        assert res.passed is False
        assert res.ending_complete == "FAIL"
        assert res.ending_natural == "FAIL"
        assert any("unfinished sentence" in b.lower() or "terputus" in b.lower() for b in res.blocking_reasons)

    def test_orchestrator_archives_qc_failure_to_failed_dir(self, tmp_path: Path):
        """Case C4: Orchestrator archives QC failure to failed/ directory with metadata."""
        temp_db = StorageRepository(db_path=str(tmp_path / "test_arch.db"))
        out_dir = tmp_path / "output"
        down_dir = tmp_path / "downloads"
        out_dir.mkdir(parents=True, exist_ok=True)
        down_dir.mkdir(parents=True, exist_ok=True)

        media = down_dir / "vid_fail.mp4"
        media.write_bytes(b"\x00" * (100 * 1024))

        meta = VideoSourceMeta(
            video_id="vid_qc_fail",
            url="https://youtube.com/watch?v=vid_qc_fail",
            title="Failed QC Video",
            duration_sec=300.0,
        )

        mock_searcher = MagicMock()
        mock_searcher.get_video_metadata.return_value = meta

        mock_filter = MagicMock()
        mock_filter.filter_video.return_value = MagicMock(is_eligible=True)

        mock_lang = MagicMock()
        mock_lang.evaluate_transcript.return_value = MagicMock(eligible=True)

        transcript = [
            TranscriptSegment(start=10.0, end=45.0, duration=35.0, text="Kalimat lengkap terucap di sini."),
        ]
        mock_tp = MagicMock()
        mock_tp.get_phrase_transcript.return_value = transcript

        cand = CandidateWindow(candidate_id="c_fail", start_sec=10.0, end_sec=45.0, duration_sec=35.0, text=transcript[0].text)
        mock_cg = MagicMock()
        mock_cg.generate_candidates.return_value = [cand]

        score = SemanticScore(
            candidate_id="c_fail",
            good_clip=True,
            score=90.0,
            hook_score=90.0,
            payoff_score=90.0,
            self_contained_score=90.0,
            reason="Good",
            suggested_start=10.0,
            suggested_end=45.0,
        )
        mock_scorer = MagicMock()
        mock_scorer.score_candidates.return_value = [score]
        mock_scorer.select_best_clip.return_value = (cand, score, "APPROVED")

        mock_refiner = BoundaryRefiner()
        mock_va = MagicMock()
        mock_va.analyze_clip.return_value = MagicMock(scene_cuts=[])
        mock_vd = MagicMock()
        mock_vd.evaluate_window.return_value = MagicMock(approved=True)
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
            notes="OK",
            preflight_mode="GEMINI_NATIVE_VIDEO",
        )

        def fake_render(video_path, output_path, edit_plan, start_sec, duration, subtitle_ass_path=None):
            Path(output_path).write_bytes(b"\x00" * (120 * 1024))
            return MagicMock(success=True)

        mock_renderer = MagicMock()
        mock_renderer.render.side_effect = fake_render

        # Local QC passes, but Gemini Video QC detects double subtitles!
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
            passed=False,
            score=50,
            double_subtitles_detected=True,
            has_double_subtitles="FAIL",
            ending_complete="PASS",
            ending_natural="PASS",
            blocking_reasons=["Double subtitles detected: source burned-in + generated"],
        )

        orch = AutoClipperOrchestrator(
            repository=temp_db,
            searcher=mock_searcher,
            source_filter=mock_filter,
            language_gate=mock_lang,
            transcript_provider=mock_tp,
            candidate_generator=mock_cg,
            semantic_scorer=mock_scorer,
            boundary_refiner=mock_refiner,
            visual_analyzer=mock_va,
            visual_director=mock_vd,
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
            res = orch.process_video(video_meta_or_url=meta, custom_video_path=str(media), dry_run=True)

        assert res.is_success is False
        assert res.status == PipelineStatus.QC_FAILED
        assert "double subtitles" in (res.error_message or "").lower()

        # Verify failed archive created under orch.failed_dir
        failed_dir = orch.failed_dir
        recent_failed = [d for d in failed_dir.iterdir() if d.is_dir() and "vid_qc_fail" in d.name]
        assert len(recent_failed) >= 1
        failed_folder = recent_failed[0]
        assert (failed_folder / "edit_plan.json").exists()
        assert (failed_folder / "qc.json").exists()
        assert (failed_folder / "clip.mp4").exists()
        assert (failed_folder / "reason.md").exists()

    def test_gemini_qc_catches_unnatural_ending_hard_fail(self, tmp_path: Path):
        """Case C3: Gemini QC catches unnatural abrupt ending (ending_natural=FAIL)."""
        dummy_clip = tmp_path / "abrupt_ending.mp4"
        dummy_clip.write_bytes(b"\x00" * (64 * 1024))

        mock_client = MagicMock()
        fake_llm_json = {
            "passed": True,
            "score": 80,
            "double_subtitles_detected": False,
            "has_double_subtitles": "PASS",
            "ending_complete": "PASS",
            "ending_natural": "FAIL",
            "ending_reason": "Ending audio cut off abruptly mid-breath without natural fade or silence.",
            "subtitle_timing": "PASS",
            "subtitle_overlap": "PASS",
            "subtitle_linger": "PASS",
            "subtitle_text_accuracy": "PASS",
            "obvious_transcription_errors": [],
            "blocking_reasons": [],
        }
        mock_client.video_completion.return_value = json.dumps(fake_llm_json)
        mock_client.extract_json.return_value = fake_llm_json

        qc = GeminiNativeVideoQC(client=mock_client)
        res = qc.evaluate_video(dummy_clip, transcript_text="Dialog contoh")

        assert res.passed is False
        assert res.ending_natural == "FAIL"
        assert any("unnatural" in b.lower() or "abrupt" in b.lower() for b in res.blocking_reasons)


# ==============================================================================
# DOMAIN 4: Orchestrator Candidate Fallback & Preflight Extension Verification
# ==============================================================================

class TestCandidateFallbackAndPreflight:
    """Verifies candidate fallback when top candidate has dangling ending and preflight extension."""

    def test_preflight_incomplete_ending_extended_successfully(self, tmp_path: Path):
        """Preflight detects incomplete ending, orchestrator extends boundary within 55s."""
        temp_db = StorageRepository(db_path=str(tmp_path / "test_preflight_ext.db"))
        out_dir = tmp_path / "output"
        down_dir = tmp_path / "downloads"
        out_dir.mkdir(parents=True, exist_ok=True)
        down_dir.mkdir(parents=True, exist_ok=True)

        media = down_dir / "vid_pf_ext.mp4"
        media.write_bytes(b"\x00" * (100 * 1024))

        meta = VideoSourceMeta(
            video_id="vid_pf_ext",
            url="https://youtube.com/watch?v=vid_pf_ext",
            title="Preflight Extension Video",
            duration_sec=300.0,
        )

        mock_searcher = MagicMock()
        mock_searcher.get_video_metadata.return_value = meta
        mock_filter = MagicMock()
        mock_filter.filter_video.return_value = MagicMock(is_eligible=True)
        mock_lang = MagicMock()
        mock_lang.evaluate_transcript.return_value = MagicMock(eligible=True)

        # Transcript:
        # 10s to 35s: initial candidate
        # 35.1s to 45.0s: completes sentence
        transcript = [
            TranscriptSegment(start=10.0, end=35.0, duration=25.0, text="Awal kalimat yang bagus."),
            TranscriptSegment(start=35.1, end=45.0, duration=9.9, text="Lanjutan kalimat sampai selesai dengan baik."),
        ]
        mock_tp = MagicMock()
        mock_tp.get_phrase_transcript.return_value = transcript

        cand = CandidateWindow(candidate_id="c_pf", start_sec=10.0, end_sec=35.0, duration_sec=25.0, text="Awal kalimat")
        mock_cg = MagicMock()
        mock_cg.generate_candidates.return_value = [cand]

        score = SemanticScore(
            candidate_id="c_pf",
            good_clip=True,
            score=90.0,
            hook_score=90.0,
            payoff_score=90.0,
            self_contained_score=90.0,
            reason="Good",
            suggested_start=10.0,
            suggested_end=35.0,
        )
        mock_scorer = MagicMock()
        mock_scorer.score_candidates.return_value = [score]
        mock_scorer.select_best_clip.return_value = (cand, score, "APPROVED")

        mock_refiner = BoundaryRefiner(min_duration_sec=20.0, max_duration_sec=55.0, laughter_buffer_sec=0.2)
        mock_va = MagicMock()
        mock_va.analyze_clip.return_value = MagicMock(scene_cuts=[])
        mock_vd = MagicMock()
        mock_vd.evaluate_window.return_value = MagicMock(approved=True)
        mock_framing = MagicMock()
        mock_framing.analyze_framing.return_value = MagicMock(layout="SAFE_WIDE", crop_windows=[])

        # Preflight flags incomplete ending!
        mock_preflight = MagicMock()
        mock_preflight.preflight_clip.return_value = VisualPreflightResult(
            usable=True,
            has_subtitles=False,
            existing_visible_subtitles=False,
            subtitle_confidence=0.99,
            ending_complete=False,
            ending_natural=False,
            ending_reason="Mid-sentence pause detected",
            blocking_issues=["Incomplete sentence ending: mid-sentence pause"],
            notes="Gemini detected incomplete ending",
            preflight_mode="GEMINI_NATIVE_VIDEO",
        )

        rendered_durations = []
        def fake_render(video_path, output_path, edit_plan, start_sec, duration, subtitle_ass_path=None):
            rendered_durations.append(duration)
            Path(output_path).write_bytes(b"\x00" * (120 * 1024))
            return MagicMock(success=True)

        mock_renderer = MagicMock()
        mock_renderer.render.side_effect = fake_render

        mock_qc = MagicMock()
        mock_qc.evaluate.return_value = ThreeTierQCReport(
            passed=True,
            publishable=True,
            technical=TechnicalQCResult(passed=True, duration=35.2),
            visual=VisualQCResult(passed=True),
            perceptual=PerceptualQCResult(passed=True, publishable=True, score=90),
        )

        mock_gemini_qc = MagicMock()
        mock_gemini_qc.evaluate_video.return_value = GeminiVideoQCResult(
            passed=True, score=90, double_subtitles_detected=False, ending_complete="PASS", ending_natural="PASS"
        )

        orch = AutoClipperOrchestrator(
            repository=temp_db,
            searcher=mock_searcher,
            source_filter=mock_filter,
            language_gate=mock_lang,
            transcript_provider=mock_tp,
            candidate_generator=mock_cg,
            semantic_scorer=mock_scorer,
            boundary_refiner=mock_refiner,
            visual_analyzer=mock_va,
            visual_director=mock_vd,
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
            res = orch.process_video(video_meta_or_url=meta, custom_video_path=str(media), dry_run=True)

        assert res.is_success is True
        assert res.status == PipelineStatus.COMPLETED
        # Clip was extended to include 45.0 + 0.2 = 45.2s (duration = 35.2s)
        assert len(rendered_durations) == 1
        assert rendered_durations[0] == pytest.approx(35.2, 0.1)

    def test_preflight_incomplete_ending_rejected_when_extension_exceeds_55s(self, tmp_path: Path):
        """Preflight detects incomplete ending, extension exceeds 55s -> REJECTED_VISUAL."""
        temp_db = StorageRepository(db_path=str(tmp_path / "test_pf_rej.db"))
        out_dir = tmp_path / "output"
        down_dir = tmp_path / "downloads"
        out_dir.mkdir(parents=True, exist_ok=True)
        down_dir.mkdir(parents=True, exist_ok=True)

        media = down_dir / "vid_pf_rej.mp4"
        media.write_bytes(b"\x00" * (100 * 1024))

        meta = VideoSourceMeta(
            video_id="vid_pf_rej",
            url="https://youtube.com/watch?v=vid_pf_rej",
            title="Preflight Rejection Video",
            duration_sec=300.0,
        )

        mock_searcher = MagicMock()
        mock_searcher.get_video_metadata.return_value = meta
        mock_filter = MagicMock()
        mock_filter.filter_video.return_value = MagicMock(is_eligible=True)
        mock_lang = MagicMock()
        mock_lang.evaluate_transcript.return_value = MagicMock(eligible=True)

        # Transcript: 10s to 45s (35s). Next segment extends to 70s (dur=60s > 55s limit)
        transcript = [
            TranscriptSegment(start=10.0, end=45.0, duration=35.0, text="Awal kalimat yang tuntas"),
            TranscriptSegment(start=45.1, end=70.0, duration=24.9, text="Kelanjutan yang terlalu panjang."),
        ]
        mock_tp = MagicMock()
        mock_tp.get_phrase_transcript.return_value = transcript

        cand = CandidateWindow(candidate_id="c_pf_rej", start_sec=10.0, end_sec=45.0, duration_sec=35.0, text="Awal kalimat")
        mock_cg = MagicMock()
        mock_cg.generate_candidates.return_value = [cand]

        score = SemanticScore(
            candidate_id="c_pf_rej",
            good_clip=True,
            score=90.0,
            hook_score=90.0,
            payoff_score=90.0,
            self_contained_score=90.0,
            reason="Good",
            suggested_start=10.0,
            suggested_end=45.0,
        )
        mock_scorer = MagicMock()
        mock_scorer.score_candidates.return_value = [score]
        mock_scorer.select_best_clip.return_value = (cand, score, "APPROVED")

        mock_refiner = BoundaryRefiner(min_duration_sec=30.0, max_duration_sec=55.0)
        mock_va = MagicMock()
        mock_va.analyze_clip.return_value = MagicMock(scene_cuts=[])
        mock_vd = MagicMock()
        mock_vd.evaluate_window.return_value = MagicMock(approved=True)
        mock_framing = MagicMock()
        mock_framing.analyze_framing.return_value = MagicMock(layout="SAFE_WIDE", crop_windows=[])

        # Preflight flags incomplete ending!
        mock_preflight = MagicMock()
        mock_preflight.preflight_clip.return_value = VisualPreflightResult(
            usable=True,
            has_subtitles=False,
            existing_visible_subtitles=False,
            subtitle_confidence=0.99,
            ending_complete=False,
            ending_natural=False,
            ending_reason="Spoke half word at clip cutoff",
            blocking_issues=["Incomplete sentence ending: Spoke half word"],
            notes="Gemini detected incomplete ending",
            preflight_mode="GEMINI_NATIVE_VIDEO",
        )

        orch = AutoClipperOrchestrator(
            repository=temp_db,
            searcher=mock_searcher,
            source_filter=mock_filter,
            language_gate=mock_lang,
            transcript_provider=mock_tp,
            candidate_generator=mock_cg,
            semantic_scorer=mock_scorer,
            boundary_refiner=mock_refiner,
            visual_analyzer=mock_va,
            visual_director=mock_vd,
            framing=mock_framing,
            visual_preflight=mock_preflight,
            output_dir=out_dir,
            download_dir=down_dir,
        )

        res = orch.process_video(video_meta_or_url=meta, custom_video_path=str(media), dry_run=True)

        assert res.is_success is False
        assert res.status == PipelineStatus.REJECTED_VISUAL
        assert "ending" in (res.rejection_reason or "").lower()

    def test_all_candidates_fail_boundary_refinement_clean_rejection(self, tmp_path: Path):
        """Case B2.2: When all ranked candidates have dangling endings exceeding 55s, pipeline rejects cleanly."""
        temp_db = StorageRepository(db_path=str(tmp_path / "test_all_fail.db"))
        out_dir = tmp_path / "output"
        down_dir = tmp_path / "downloads"
        out_dir.mkdir(parents=True, exist_ok=True)
        down_dir.mkdir(parents=True, exist_ok=True)

        media = down_dir / "vid_all_fail.mp4"
        media.write_bytes(b"\x00" * (100 * 1024))

        meta = VideoSourceMeta(
            video_id="vid_all_fail",
            url="https://youtube.com/watch?v=vid_all_fail",
            title="All Fail Video",
            duration_sec=300.0,
        )

        mock_searcher = MagicMock()
        mock_searcher.get_video_metadata.return_value = meta
        mock_filter = MagicMock()
        mock_filter.filter_video.return_value = MagicMock(is_eligible=True)
        mock_lang = MagicMock()
        mock_lang.evaluate_transcript.return_value = MagicMock(eligible=True)

        # Both candidates end with dangling phrases that extend > 55s
        transcript = [
            TranscriptSegment(start=10.0, end=45.0, duration=35.0, text="Awal cerita dan kemudian waktu itu masih"),
            TranscriptSegment(start=45.1, end=70.0, duration=24.9, text="Kelanjutan panjang sekali."),
            TranscriptSegment(start=100.0, end=135.0, duration=35.0, text="Cerita kedua karena sebenarnya"),
            TranscriptSegment(start=135.1, end=160.0, duration=24.9, text="Kelanjutan kedua panjang sekali."),
        ]
        mock_tp = MagicMock()
        mock_tp.get_phrase_transcript.return_value = transcript

        cand1 = CandidateWindow(candidate_id="c1", start_sec=10.0, end_sec=45.0, duration_sec=35.0, text="c1")
        cand2 = CandidateWindow(candidate_id="c2", start_sec=100.0, end_sec=135.0, duration_sec=35.0, text="c2")
        mock_cg = MagicMock()
        mock_cg.generate_candidates.return_value = [cand1, cand2]

        score1 = SemanticScore(
            candidate_id="c1", good_clip=True, score=90.0, hook_score=90.0, payoff_score=90.0, self_contained_score=90.0,
            reason="Good", suggested_start=10.0, suggested_end=45.0,
        )
        score2 = SemanticScore(
            candidate_id="c2", good_clip=True, score=85.0, hook_score=85.0, payoff_score=85.0, self_contained_score=85.0,
            reason="Good", suggested_start=100.0, suggested_end=135.0,
        )
        mock_scorer = MagicMock()
        mock_scorer.score_candidates.return_value = [score1, score2]
        mock_scorer.select_best_clip.return_value = (cand1, score1, "APPROVED")

        mock_refiner = BoundaryRefiner(min_duration_sec=30.0, max_duration_sec=55.0)

        orch = AutoClipperOrchestrator(
            repository=temp_db,
            searcher=mock_searcher,
            source_filter=mock_filter,
            language_gate=mock_lang,
            transcript_provider=mock_tp,
            candidate_generator=mock_cg,
            semantic_scorer=mock_scorer,
            boundary_refiner=mock_refiner,
            output_dir=out_dir,
            download_dir=down_dir,
        )

        res = orch.process_video(video_meta_or_url=meta, custom_video_path=str(media), dry_run=True)

        assert res.is_success is False
        assert res.status == PipelineStatus.NO_GOOD_CLIP
        assert "boundary refinement" in (res.rejection_reason or "").lower()


# ==============================================================================
# DOMAIN 5: Boundary Invariant Fuzzing & Numerical Bounds Check
# ==============================================================================

class TestBoundaryInvariants:
    """Verifies strict adherence to 30.0s - 55.0s boundaries under random/adversarial inputs."""

    def test_extension_never_exceeds_max_duration(self):
        """Fuzz check: extend_to_sentence_boundary never returns duration > max_duration_sec."""
        refiner = BoundaryRefiner(min_duration_sec=30.0, max_duration_sec=55.0, laughter_buffer_sec=0.2)

        import random
        random.seed(42)

        for _ in range(50):
            start = random.uniform(0.0, 100.0)
            initial_end = start + random.uniform(20.0, 54.0)

            # Generate random segments
            curr = start
            segments = []
            while curr < start + 80.0:
                seg_dur = random.uniform(1.0, 10.0)
                seg_end = curr + seg_dur
                # Randomly pick complete or dangling
                dangling = random.choice([True, False])
                text = "Suatu kalimat dan" if dangling else "Suatu kalimat yang sudah lengkap."
                segments.append(TranscriptSegment(start=curr, end=seg_end, duration=seg_dur, text=text))
                curr = seg_end + random.uniform(0.1, 0.5)

            can_ext, new_end, reason = refiner.extend_to_sentence_boundary(
                start_sec=start,
                end_sec=initial_end,
                segments=segments,
                max_duration_sec=55.0,
            )

            if can_ext:
                dur = round(new_end - start, 2)
                assert dur <= 55.0, f"Invariant violated: duration {dur}s > 55.0s (start={start}, new_end={new_end})"
            else:
                assert new_end == initial_end

