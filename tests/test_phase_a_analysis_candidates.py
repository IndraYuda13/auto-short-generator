"""Unit tests for Phase A (Part 2): Candidate Generator & Boundary Refiner.

Covers:
1. Candidate Generator (Blueprint Stage 4):
   - RawCandidate Pydantic model (candidate_id, start_sec, end_sec, duration_sec, text, segments, pre_context, post_context).
   - Window duration 25–70 seconds.
   - Sentence boundaries (. ? !) & natural pause boundaries (gap >= 0.5s).
   - Pre-context (1-2 sentences prior) and post-context (1-2 sentences after).
   - Sliding window traversal and deduplication.
   - Backward compatibility with CandidateWindow.

2. Boundary Refiner (Blueprint Stage 7):
   - RefinementResult Pydantic model (is_valid, refined_start, refined_end, duration, rejection_reason).
   - Start boundary snapping before hook avoiding mid-word cuts and scene-cut jitter.
   - End boundary snapping after payoff with natural breathing room (0.1–0.3s) preserving reaction/laughter.
   - Strict 30–55s duration enforcement.
   - Rejection when candidate cannot fit within [30.0, 55.0] seconds.
   - Backward compatibility with RefinedBoundaryResult.
"""

import pytest
from typing import List

from transcription.transcript_provider import TranscriptSegment
from transcription.whisper_aligner import WordToken
from analysis.candidate_generator import CandidateGenerator, RawCandidate, CandidateWindow
from analysis.boundary_refiner import BoundaryRefiner, RefinementResult, RefinedBoundaryResult


# ==============================================================================
# 1. CANDIDATE GENERATOR TESTS
# ==============================================================================

def test_raw_candidate_model_schema():
    """Verify RawCandidate model attributes, types, and backward compatibility."""
    seg_dict = {"start": 10.0, "end": 14.5, "duration": 4.5, "text": "Ini kalimat pembuka.", "source": "yt"}
    cand = RawCandidate(
        candidate_id="cand_01",
        start_sec=10.0,
        end_sec=42.5,
        duration_sec=32.5,
        text="Ini kalimat pembuka dan lanjutannya.",
        segments=[seg_dict],
        pre_context="Konteks sebelum kandidat.",
        post_context="Konteks sesudah kandidat."
    )

    assert cand.candidate_id == "cand_01"
    assert cand.start_sec == 10.0
    assert cand.end_sec == 42.5
    assert cand.duration_sec == 32.5
    assert cand.duration == 32.5  # Backward compatibility property
    assert cand.text == "Ini kalimat pembuka dan lanjutannya."
    assert len(cand.segments) == 1
    assert isinstance(cand.segments[0], dict)
    assert cand.pre_context == "Konteks sebelum kandidat."
    assert cand.post_context == "Konteks sesudah kandidat."


def test_candidate_generator_duration_bounds_25_70s():
    """Verify that all generated candidates strictly fall within 25–70s."""
    generator = CandidateGenerator(min_duration_sec=25.0, max_duration_sec=70.0, min_pause_sec=0.5)

    segments = [
        TranscriptSegment(start=0.0, end=4.5, duration=4.5, text="Halo teman-teman semua."),
        TranscriptSegment(start=5.1, end=10.0, duration=4.9, text="Hari ini kita akan membahas tentang AI."),  # 0.6s pause
        TranscriptSegment(start=10.2, end=18.0, duration=7.8, text="Banyak orang mengira AI akan menggantikan programmer."),
        TranscriptSegment(start=18.2, end=25.0, duration=6.8, text="Padahal AI justru menjadi asisten super produktif."),
        TranscriptSegment(start=25.2, end=33.0, duration=7.8, text="Contohnya auto short generator ini yang sangat canggih."),
        TranscriptSegment(start=33.8, end=40.0, duration=6.2, text="Kamu bisa memotong podcast otomatis dalam hitungan menit."),  # 0.8s pause
        TranscriptSegment(start=40.2, end=48.0, duration=7.8, text="Kuncinya adalah seleksi hook dan payoff yang kuat."),
        TranscriptSegment(start=48.2, end=55.0, duration=6.8, text="Jadi mulailah beradaptasi hari ini juga!"),
        TranscriptSegment(start=56.0, end=62.0, duration=6.0, text="Jangan tunggu sampai terlambat teman-teman."),  # 1.0s pause
        TranscriptSegment(start=62.5, end=70.0, duration=7.5, text="Di video berikutnya kita akan bahas backend system-nya."),
    ]

    candidates = generator.generate_candidates(segments)
    assert len(candidates) >= 1

    for cand in candidates:
        assert 25.0 <= cand.duration_sec <= 70.0
        assert 25.0 <= cand.duration <= 70.0
        assert len(cand.segments) >= 2
        assert isinstance(cand.segments[0], dict)
        assert cand.text != ""


def test_candidate_generator_sentence_and_pause_boundaries():
    """Verify that sentence punctuation (. ? !) and gaps >= 0.5s trigger boundary detection."""
    generator = CandidateGenerator(min_duration_sec=25.0, max_duration_sec=70.0, min_pause_sec=0.5)

    segments = [
        TranscriptSegment(start=0.0, end=6.0, duration=6.0, text="Pembukaan podcast tanpa titik"),
        TranscriptSegment(start=6.8, end=12.0, duration=5.2, text="Ada jeda 0.8 detik sebelumnya!"),  # gap=0.8s, ends with !
        TranscriptSegment(start=12.1, end=20.0, duration=7.9, text="Pertanyaan penting: apakah ini berhasil?"),  # ends with ?
        TranscriptSegment(start=20.1, end=28.0, duration=7.9, text="Tentu saja berhasil dengan sangat baik."),  # ends with .
        TranscriptSegment(start=28.2, end=35.0, duration=6.8, text="Kesimpulan sesi pertama selesai."),  # ends with .
        TranscriptSegment(start=36.0, end=42.0, duration=6.0, text="Ada jeda 1.0 detik sebelum segmen ini."),  # gap=1.0s
    ]

    candidates = generator.generate_candidates(segments)
    assert len(candidates) >= 1

    cand0 = candidates[0]
    assert cand0.start_sec == 0.0
    assert cand0.duration_sec >= 25.0
    assert cand0.has_pause_before is True  # start of video is considered natural start


def test_candidate_generator_pre_and_post_context():
    """Verify pre_context (1-2 preceding sentences) and post_context (1-2 following sentences)."""
    generator = CandidateGenerator(min_duration_sec=25.0, max_duration_sec=45.0, min_pause_sec=0.5)

    segments = [
        TranscriptSegment(start=0.0, end=5.0, duration=5.0, text="Selamat datang di podcast."),
        TranscriptSegment(start=5.5, end=10.0, duration=4.5, text="Ini adalah episode kedua kita."),  # pause 0.5s
        # Target candidate start around here (idx 2, 10.5s)
        TranscriptSegment(start=10.5, end=18.0, duration=7.5, text="Cerita bermula saat saya merantau ke Jakarta."),
        TranscriptSegment(start=18.2, end=26.0, duration=7.8, text="Saya tidak punya kenalan satupun di kota ini."),
        TranscriptSegment(start=26.2, end=34.0, duration=7.8, text="Tidur di kosan sempit dan makan mie instan tiap hari."),
        TranscriptSegment(start=34.2, end=42.0, duration=7.8, text="Tapi tekad saya untuk sukses jauh lebih besar."),
        # Candidate ends around here (idx 5, 42.0s)
        TranscriptSegment(start=42.8, end=48.0, duration=5.2, text="Itulah yang membuat saya bertahan sampai sekarang."),
        TranscriptSegment(start=48.5, end=55.0, duration=6.5, text="Mari kita lanjut ke pembahasan berikutnya."),
        TranscriptSegment(start=55.5, end=62.0, duration=6.5, text="Kita punya bintang tamu spesial hari ini."),
        TranscriptSegment(start=62.5, end=70.0, duration=7.5, text="Beliau adalah founder dari startup teknologi ternama."),
    ]

    candidates = generator.generate_candidates(segments)
    assert len(candidates) >= 1

    # Find a candidate that starts at or after segment 2
    cand_mid = next((c for c in candidates if c.start_sec >= 10.0), None)
    if cand_mid:
        # Pre-context should contain sentences from segments 0 and 1
        assert cand_mid.pre_context != ""
        assert "Selamat datang" in cand_mid.pre_context or "episode kedua" in cand_mid.pre_context

        # Post-context should contain sentences from following segments
        assert cand_mid.post_context != ""
        assert "bintang tamu" in cand_mid.post_context or "startup" in cand_mid.post_context

    # First candidate at 0.0s must have empty pre_context
    cand_first = next((c for c in candidates if c.start_sec == 0.0), None)
    if cand_first:
        assert cand_first.pre_context == ""


def test_candidate_generator_empty_input():
    """Verify graceful handling of empty segments list."""
    generator = CandidateGenerator()
    assert generator.generate_candidates([]) == []


def test_candidate_window_backward_compatibility():
    """Verify CandidateWindow alias accepts legacy keyword arguments and functions identically."""
    cw = CandidateWindow(
        candidate_id="cw_test",
        start_sec=5.0,
        end_sec=40.0,
        duration_sec=35.0,
        text="Legacy compatibility test text.",
        segment_count=3,
        segments=[
            TranscriptSegment(start=5.0, end=15.0, duration=10.0, text="Satu."),
            TranscriptSegment(start=15.0, end=25.0, duration=10.0, text="Dua."),
            TranscriptSegment(start=25.0, end=40.0, duration=15.0, text="Tiga."),
        ]
    )
    assert cw.candidate_id == "cw_test"
    assert cw.duration_sec == 35.0
    assert cw.duration == 35.0
    assert cw.segment_count == 3
    assert len(cw.segments) == 3
    assert isinstance(cw.segments[0], dict)


# ==============================================================================
# 2. BOUNDARY REFINER TESTS
# ==============================================================================

def test_refinement_result_model_schema():
    """Verify RefinementResult fields, types, and backward compatibility properties."""
    res = RefinementResult(
        is_valid=True,
        refined_start=12.4,
        refined_end=48.6,
        duration=36.2,
        rejection_reason=None,
        snapped_to_scene_cut=True,
        laughter_buffer_added=0.2
    )
    assert res.is_valid is True
    assert res.accepted is True
    assert res.refined_start == 12.4
    assert res.start_sec == 12.4
    assert res.refined_end == 48.6
    assert res.end_sec == 48.6
    assert res.duration == 36.2
    assert res.rejection_reason is None
    assert res.reason == ""
    assert res.snapped_to_scene_cut is True
    assert res.laughter_buffer_added == 0.2


def test_boundary_refiner_snap_start_avoid_mid_word():
    """Verify that start boundary snaps before word start when Gemini suggests mid-word cut."""
    refiner = BoundaryRefiner(min_duration_sec=30.0, max_duration_sec=55.0, laughter_buffer_sec=0.2)

    word_tokens = [
        WordToken(word="Waktu", start=10.0, end=10.4),
        WordToken(word="gue", start=10.5, end=10.8),
        WordToken(word="nekat", start=10.9, end=11.4),
        WordToken(word="mulai", start=11.5, end=11.9),
        WordToken(word="bisnis", start=12.0, end=12.6),
        # Final words around 45s
        WordToken(word="hasilnya", start=44.0, end=44.6),
        WordToken(word="sukses", start=44.7, end=45.2),
    ]

    segments = [
        TranscriptSegment(start=10.0, end=20.0, duration=10.0, text="Waktu gue nekat mulai bisnis"),
        TranscriptSegment(start=20.0, end=45.5, duration=25.5, text="prosesnya panjang dan hasilnya sukses."),
    ]

    # Gemini suggests initial_start=11.1 (which cuts directly inside word 'nekat' [10.9, 11.4])
    res = refiner.refine_boundaries(
        initial_start=11.1,
        initial_end=45.2,
        phrase_segments=segments,
        word_tokens=word_tokens,
    )

    assert res.is_valid is True
    # Start should snap before 'nekat' (10.9 - 0.05 = 10.85s)
    assert res.refined_start == 10.85
    assert 30.0 <= res.duration <= 55.0


def test_boundary_refiner_snap_start_scene_cut():
    """Verify that start boundary snaps to adjacent scene transition cut (within 0.4s)."""
    refiner = BoundaryRefiner(min_duration_sec=30.0, max_duration_sec=55.0, laughter_buffer_sec=0.2)

    segments = [
        TranscriptSegment(start=15.0, end=30.0, duration=15.0, text="Pembuka segmen."),
        TranscriptSegment(start=30.2, end=50.0, duration=19.8, text="Penutup segmen."),
    ]

    # Scene cut exists at 14.8s (within 0.4s of initial_start=15.0)
    res = refiner.refine_boundaries(
        initial_start=15.0,
        initial_end=50.0,
        phrase_segments=segments,
        scene_cuts=[14.8, 30.0]
    )

    assert res.is_valid is True
    assert res.snapped_to_scene_cut is True
    assert res.refined_start == 14.8


def test_boundary_refiner_snap_end_after_payoff_breathing_room():
    """Verify that end boundary preserves sentence completion and adds natural breathing room (0.2s)."""
    refiner = BoundaryRefiner(min_duration_sec=30.0, max_duration_sec=55.0, laughter_buffer_sec=0.2)

    segments = [
        TranscriptSegment(start=5.0, end=20.0, duration=15.0, text="Awal kisah perjuangan."),
        TranscriptSegment(start=20.2, end=42.0, duration=21.8, text="Akhirnya semua terbayar lunas dan semua orang tertawa."),
    ]

    res = refiner.refine_boundaries(
        initial_start=5.0,
        initial_end=42.0,
        phrase_segments=segments
    )

    assert res.is_valid is True
    # End should snap to 42.0 + 0.2 = 42.2s
    assert res.refined_end == 42.2
    assert res.laughter_buffer_added == 0.2
    assert res.duration == pytest.approx(37.2, 0.05)


def test_boundary_refiner_enforce_strict_duration_reject_short():
    """Verify strict duration enforcement: reject candidate < 30s."""
    refiner = BoundaryRefiner(min_duration_sec=30.0, max_duration_sec=55.0)

    segments = [
        TranscriptSegment(start=10.0, end=18.0, duration=8.0, text="Kalimat pertama."),
        TranscriptSegment(start=18.2, end=25.0, duration=6.8, text="Kalimat kedua."),
    ]

    # Max duration is 15s, below 30s min
    res = refiner.refine_boundaries(
        initial_start=10.0,
        initial_end=25.0,
        phrase_segments=segments
    )

    assert res.is_valid is False
    assert res.accepted is False
    assert res.rejection_reason is not None
    assert "below minimum target 30.0s" in res.rejection_reason
    assert res.duration < 30.0


def test_boundary_refiner_enforce_strict_duration_reject_long():
    """Verify strict duration enforcement: reject candidate > 55s."""
    refiner = BoundaryRefiner(min_duration_sec=30.0, max_duration_sec=55.0)

    segments = [
        TranscriptSegment(start=0.0, end=30.0, duration=30.0, text="Bagian pertama podcast yang panjang."),
        TranscriptSegment(start=30.5, end=65.0, duration=34.5, text="Bagian kedua podcast yang juga sangat panjang."),
    ]

    res = refiner.refine_boundaries(
        initial_start=0.0,
        initial_end=65.0,
        phrase_segments=segments
    )

    assert res.is_valid is False
    assert res.accepted is False
    assert res.rejection_reason is not None
    assert "exceeds maximum target 55.0s" in res.rejection_reason
    assert res.duration > 55.0


def test_boundary_refiner_invalid_timestamps():
    """Verify rejection when initial_end <= initial_start."""
    refiner = BoundaryRefiner()
    res = refiner.refine_boundaries(
        initial_start=45.0,
        initial_end=40.0,
        phrase_segments=[]
    )
    assert res.is_valid is False
    assert res.rejection_reason is not None
    assert "end <= start" in res.rejection_reason
    assert res.duration == 0.0


def test_boundary_refiner_micro_adjustment_extension():
    """Verify that a candidate slightly under 30s can be cleanly extended to next phrase."""
    refiner = BoundaryRefiner(min_duration_sec=30.0, max_duration_sec=55.0, laughter_buffer_sec=0.2)

    segments = [
        TranscriptSegment(start=10.0, end=22.0, duration=12.0, text="Kalimat pembuka hook."),
        TranscriptSegment(start=22.2, end=38.0, duration=15.8, text="Penjelasan inti."),  # 10 to 38.2 = 28.2s (< 30s)
        TranscriptSegment(start=38.4, end=45.0, duration=6.6, text="Kesimpulan yang pas."),   # 10 to 45.2 = 35.2s (in 30-55s)
    ]

    # Initial suggest stops at 38.0s (duration ~28s, below 30s)
    res = refiner.refine_boundaries(
        initial_start=10.0,
        initial_end=38.0,
        phrase_segments=segments
    )

    # Should extend to include segment 2 (end at 45.0 + 0.2 = 45.2s)
    assert res.is_valid is True
    assert 30.0 <= res.duration <= 55.0
    assert res.refined_end == 45.2
