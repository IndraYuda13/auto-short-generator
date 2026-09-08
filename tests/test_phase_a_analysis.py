"""Unit tests for Phase A Analysis package:
- candidate_generator
- semantic_scorer
- visual_analyzer
- visual_director
- boundary_refiner
"""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from pathlib import Path

from transcription.transcript_provider import TranscriptSegment
from transcription.whisper_aligner import WordToken
from analysis.candidate_generator import CandidateGenerator, CandidateWindow
from analysis.semantic_scorer import SemanticScorer, SemanticScore
from analysis.visual_analyzer import VisualAnalyzer, VisualAnalysisReport
from analysis.visual_director import VisualDirector, VisualDirectorVerdict
from analysis.boundary_refiner import BoundaryRefiner, RefinedBoundaryResult


# --- 1. CandidateGenerator Tests ---

def test_candidate_generator_duration_and_pause_boundaries():
    """Candidate generator groups segments into 25-70s windows respecting pause boundaries."""
    generator = CandidateGenerator(min_duration_sec=25.0, max_duration_sec=70.0, min_pause_sec=0.5)

    segments = [
        TranscriptSegment(start=0.0, end=5.0, duration=5.0, text="Halo semua selamat datang di podcast."),
        TranscriptSegment(start=5.6, end=10.0, duration=4.4, text="Hari ini kita mau ngobrol santai."),  # 0.6s pause before
        TranscriptSegment(start=10.2, end=18.0, duration=7.8, text="Banyak orang bilang kalau mulai bisnis itu susah."),
        TranscriptSegment(start=18.1, end=26.0, duration=7.9, text="Tapi sebenarnya yang paling penting adalah konsistensi."),
        TranscriptSegment(start=26.2, end=35.0, duration=8.8, text="Dan jangan takut untuk mencoba hal baru."),
        TranscriptSegment(start=36.0, end=42.0, duration=6.0, text="Itu kunci sukses yang gue pelajari selama 10 tahun."),  # 1.0s pause before
        TranscriptSegment(start=43.0, end=48.0, duration=5.0, text="Topik berikutnya adalah tentang pendanaan."),
    ]

    candidates = generator.generate_candidates(segments)
    assert len(candidates) >= 1

    for cand in candidates:
        assert 25.0 <= cand.duration <= 70.0
        assert len(cand.segments) >= 2
        assert cand.text.strip() != ""


# --- 2. SemanticScorer Tests ---

def test_semantic_scorer_success_good_clip():
    """Semantic scorer parses structured JSON and marks good_clip when criteria met."""
    scorer = SemanticScorer()

    cand = CandidateWindow(
        candidate_id="cand_01",
        start_sec=10.0,
        end_sec=50.0,
        duration=40.0,
        text="Gue waktu itu bangkrut total tapi bangkit lagi.",
        segment_count=5
    )

    mock_llm_json = """{
      "candidate_id": "cand_01",
      "good_clip": true,
      "score": 88.0,
      "hook_score": 90.0,
      "payoff_score": 85.0,
      "self_contained_score": 92.0,
      "reason": "Cerita bangkrut yang sangat self-contained dengan hook kuat.",
      "suggested_start": 10.5,
      "suggested_end": 49.0
    }"""

    with patch.object(scorer, "_call_llm", return_value=mock_llm_json):
        result = scorer.score_candidate(cand, video_title="Podcast Inspiratif")
        assert result.good_clip is True
        assert result.score == 88.0
        assert result.hook_score == 90.0
        assert result.self_contained_score == 92.0
        assert result.suggested_start == 10.5
        assert result.suggested_end == 49.0


def test_semantic_scorer_no_good_clip_found():
    """Returns 'NO GOOD CLIP FOUND' when all candidates fail criteria."""
    scorer = SemanticScorer()

    c1 = CandidateWindow(candidate_id="c1", start_sec=0.0, end_sec=35.0, duration=35.0, text="Teks satu", segment_count=3)
    c2 = CandidateWindow(candidate_id="c2", start_sec=40.0, end_sec=75.0, duration=35.0, text="Teks dua", segment_count=3)

    s1 = SemanticScore(
        candidate_id="c1", good_clip=False, score=55.0, hook_score=50.0,
        payoff_score=50.0, self_contained_score=40.0, reason="Tidak self-contained",
        suggested_start=0.0, suggested_end=35.0
    )
    s2 = SemanticScore(
        candidate_id="c2", good_clip=False, score=62.0, hook_score=60.0,
        payoff_score=58.0, self_contained_score=55.0, reason="Konteks menggantung",
        suggested_start=40.0, suggested_end=75.0
    )

    best_cand, best_score, status = scorer.select_best_clip([c1, c2], [s1, s2])
    assert best_cand is None
    assert best_score is None
    assert status == "NO GOOD CLIP FOUND"


# --- 3. VisualAnalyzer Tests ---

def test_visual_analyzer_rejection_on_blank_frames():
    """VisualAnalyzer rejects when blank frames exceed threshold."""
    analyzer = VisualAnalyzer()
    analyzer.MAX_BLANK_FRAME_RATIO = 0.10

    # Mock video frames with black frames
    black_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    sampled = [(float(i), black_frame) for i in range(10)]

    # Mock analyze_window flow
    report = VisualAnalysisReport(
        video_path="dummy.mp4",
        start_sec=0.0,
        end_sec=30.0,
        duration=30.0,
        blank_frame_ratio=0.50,
        subject_presence_ratio=0.90,
        is_viable=False,
        rejection_reasons=["Too many blank/black frames (50.0% > 10%)"]
    )
    assert report.is_viable is False
    assert any("blank" in r.lower() for r in report.rejection_reasons)


# --- 4. VisualDirector Tests ---

def test_visual_director_rejects_chaotic_continuity():
    """VisualDirector rejects when continuity risk is high or subject missing."""
    director = VisualDirector()

    report = VisualAnalysisReport(
        video_path="/dummy/video.mp4",
        start_sec=0.0,
        end_sec=35.0,
        duration=35.0,
        scene_cuts=[2.0, 4.0, 6.0, 8.0, 10.0, 12.0],
        scene_cut_rate_per_sec=0.17,
        blank_frame_ratio=0.0,
        subject_presence_ratio=0.85,
        is_viable=True
    )

    mock_gemini_json = """{
      "approved": false,
      "confidence": 0.95,
      "shot_type": "broll_or_graphic",
      "recommended_framing": "BLURRED_FALLBACK",
      "faces_visible": 0,
      "subject_visible_ratio": 0.40,
      "continuity_risk": "high",
      "rejection_reasons": ["Subject missing >30% duration", "High visual continuity risk between scene cuts"],
      "notes": "Gambar tidak konsisten dan pembicara menghilang."
    }"""

    with patch.object(director, "extract_representative_frames_b64", return_value=["base64_sample"]):
        with patch.object(director, "_extract_json", return_value={
            "approved": False,
            "confidence": 0.95,
            "shot_type": "broll_or_graphic",
            "recommended_framing": "BLURRED_FALLBACK",
            "faces_visible": 0,
            "subject_visible_ratio": 0.40,
            "continuity_risk": "high",
            "rejection_reasons": ["Subject missing >30% duration", "High visual continuity risk between scene cuts"],
            "notes": "Gambar tidak konsisten"
        }):
            with patch("requests.post") as mock_post:
                mock_post.return_value.status_code = 200
                mock_post.return_value.json.return_value = {
                    "choices": [{"message": {"content": mock_gemini_json}}]
                }
                verdict = director.direct_clip("/dummy/video.mp4", report)
                assert verdict.approved is False
                assert verdict.continuity_risk == "high"
                assert len(verdict.rejection_reasons) >= 1


# --- 5. BoundaryRefiner Tests ---

def test_boundary_refiner_snaps_boundaries_and_enforces_30_55s():
    """BoundaryRefiner snaps to speech and enforces 30-55s duration."""
    refiner = BoundaryRefiner(min_duration_sec=30.0, max_duration_sec=55.0, laughter_buffer_sec=0.5)

    segments = [
        TranscriptSegment(start=10.0, end=15.0, duration=5.0, text="Hook pembuka kalimat."),
        TranscriptSegment(start=15.2, end=25.0, duration=9.8, text="Penjelasan inti di tengah cerita."),
        TranscriptSegment(start=25.5, end=48.0, duration=22.5, text="Dan akhirnya ini kesimpulan yang sangat lucu."),
        TranscriptSegment(start=48.5, end=55.0, duration=6.5, text="Kalimat penutup berikutnya."),
    ]

    # Initial candidate suggested range: [10.0, 48.0] (duration ~38s)
    res = refiner.refine_boundaries(
        initial_start=10.0,
        initial_end=48.0,
        phrase_segments=segments,
        scene_cuts=[9.8, 25.0]  # Scene cut at 9.8s near start (within 0.4s)
    )

    assert res.accepted is True
    assert 30.0 <= res.duration <= 55.0
    assert res.snapped_to_scene_cut is True
    assert res.start_sec == 9.8  # Snapped to scene cut
    assert res.laughter_buffer_added == 0.5


def test_boundary_refiner_rejects_duration_too_short():
    """BoundaryRefiner rejects candidate if duration cannot reach 30 seconds."""
    refiner = BoundaryRefiner(min_duration_sec=30.0, max_duration_sec=55.0)

    segments = [
        TranscriptSegment(start=5.0, end=10.0, duration=5.0, text="Satu kalimat."),
        TranscriptSegment(start=10.2, end=18.0, duration=7.8, text="Dua kalimat."),
    ]

    # Total duration is only 13s
    res = refiner.refine_boundaries(
        initial_start=5.0,
        initial_end=18.0,
        phrase_segments=segments
    )

    assert res.accepted is False
    assert "below minimum target 30.0s" in res.reason


def test_boundary_refiner_rejects_duration_too_long():
    """BoundaryRefiner rejects candidate if duration exceeds 55 seconds."""
    refiner = BoundaryRefiner(min_duration_sec=30.0, max_duration_sec=55.0)

    segments = [
        TranscriptSegment(start=0.0, end=30.0, duration=30.0, text="Paragraf panjang."),
        TranscriptSegment(start=30.5, end=65.0, duration=34.5, text="Paragraf kedua yang sangat panjang."),
    ]

    # Total duration is ~65s
    res = refiner.refine_boundaries(
        initial_start=0.0,
        initial_end=65.0,
        phrase_segments=segments
    )

    assert res.accepted is False
    assert "exceeds maximum target 55.0s" in res.reason
