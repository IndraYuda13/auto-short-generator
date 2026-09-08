import pytest
from pacing import PacingEngine


def test_pacing_timeline_no_cuts_short_gaps():
    engine = PacingEngine(min_silence_gap=0.65, speech_padding=0.15)
    # Speech words with small gaps (<0.65s) and speech starting at 0.0s
    words = [
        {"word": "halo", "start": 0.0, "end": 1.5},
        {"word": "semua", "start": 1.8, "end": 2.2},  # gap 0.3s
        {"word": "selamat", "start": 2.5, "end": 4.0}, # gap 0.3s
    ]
    segs, remap = engine.build_pacing_timeline(words, clip_duration=4.0)

    # No cut made -> single continuous interval
    assert len(segs) == 1
    assert segs[0].dst_end == 4.0
    assert remap(2.0) == 2.0


def test_pacing_timeline_monotonic_remapping():
    engine = PacingEngine(min_silence_gap=0.65, speech_padding=0.15, max_silence_removal=1.0)
    # Speech with large dead air gap of 2.0s
    words = [
        {"word": "kalimat", "start": 1.0, "end": 2.0},
        {"word": "lanjutan", "start": 4.5, "end": 5.5}, # gap = 2.5s (from 2.0 to 4.5)
    ]
    segs, remap = engine.build_pacing_timeline(words, clip_duration=7.0)

    # Must have cut the gap
    assert len(segs) >= 2
    # Verify monotonic projection
    t_samples = [0.0, 1.0, 1.5, 2.0, 2.5, 3.5, 4.5, 5.0, 6.0, 7.0]
    remapped = [remap(t) for t in t_samples]
    for i in range(1, len(remapped)):
        assert remapped[i] >= remapped[i - 1], f"Remapping is not monotonic: {remapped}"
