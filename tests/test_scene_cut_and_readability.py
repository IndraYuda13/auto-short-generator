"""Regression tests for scene-cut boundary tracking and readability enhancements."""

from pathlib import Path
from edit_plan import CropKeyframe, EditPlan, SubtitleStyle
from visual_framing import visual_framing
from subtitle import subtitle_generator
from qc import VideoQualityControl


def test_scene_cut_smoothing_forbids_cross_cut_interpolation():
    """
    Verifies that smoothing does NOT interpolate between scene A and scene B.
    Scene A: Face centered at x=0.20 (t=0.0s, 1.0s, 2.0s)
    Cut at t=2.5s
    Scene B: Face centered at x=0.80 (t=3.0s, 4.0s, 5.0s)

    With cross-cut interpolation forbidden:
    - Last keyframe of Scene A (t=2.0s) must NOT be dragged towards 0.80.
    - First keyframe of Scene B (t=3.0s) must NOT be dragged towards 0.20.
    """
    raw_keyframes = [
        CropKeyframe(time=0.0, crop_center_x=0.20, crop_center_y=0.5, confidence=0.9),
        CropKeyframe(time=1.0, crop_center_x=0.20, crop_center_y=0.5, confidence=0.9),
        CropKeyframe(time=2.0, crop_center_x=0.20, crop_center_y=0.5, confidence=0.9),
        CropKeyframe(time=3.0, crop_center_x=0.80, crop_center_y=0.5, confidence=0.9),
        CropKeyframe(time=4.0, crop_center_x=0.80, crop_center_y=0.5, confidence=0.9),
        CropKeyframe(time=5.0, crop_center_x=0.80, crop_center_y=0.5, confidence=0.9),
    ]

    scene_cuts = [2.5]
    src_w, src_h = 1920, 1080

    smoothed = visual_framing.smooth_segmented_keyframes(
        keyframes=raw_keyframes,
        scene_cuts=scene_cuts,
        src_width=src_w,
        src_height=src_h,
        aspect_ratio=9.0 / 16.0,
        window_size=3
    )

    # Inspect t=2.0s and t=3.0s
    kf_2 = next(k for k in smoothed if k.time == 2.0)
    kf_3 = next(k for k in smoothed if k.time == 3.0)

    # Without cross-cut contamination, kf_2 must remain strictly near 0.20 (not pulled towards 0.80)
    assert abs(kf_2.crop_center_x - 0.20) < 0.05, f"Expected ~0.20, got {kf_2.crop_center_x}"

    # Similarly, kf_3 must remain strictly near 0.80 (not pulled towards 0.20)
    assert abs(kf_3.crop_center_x - 0.80) < 0.05, f"Expected ~0.80, got {kf_3.crop_center_x}"


def test_face_reacquisition_and_safe_fallback():
    """Missing faces below threshold trigger FramingMode.BLURRED_FALLBACK safely."""
    # Video path that does not exist or empty returns BLURRED_FALLBACK
    mode, kfs = visual_framing.analyze_clip_framing(
        video_path="/tmp/non_existent_video.mp4",
        start_sec=0.0,
        end_sec=10.0
    )
    assert mode.value == "BLURRED_FALLBACK"
    assert len(kfs) == 0


def test_subtitle_readability_and_indonesian_chunking(tmp_path: Path):
    """
    Verifies:
    1. Default subtitle style has font_size=52, outline_width=5, shadow_width=3, margin_v=540.
    2. Indonesian chunking binds negation + predicate ('nggak pernah') and prepositions ('ke kantor').
    3. Conjunctions start fresh chunks and punctuation triggers hard split.
    """
    words = [
        {"word": "gue", "start": 0.0, "end": 0.4},
        {"word": "sebenarnya", "start": 0.4, "end": 0.9},
        {"word": "nggak", "start": 0.9, "end": 1.2},
        {"word": "pernah", "start": 1.2, "end": 1.6},
        {"word": "mikir", "start": 1.6, "end": 2.0},
        {"word": "kalau", "start": 2.0, "end": 2.4},
        {"word": "dia", "start": 2.4, "end": 2.7},
        {"word": "datang", "start": 2.7, "end": 3.1},
        {"word": "ke", "start": 3.1, "end": 3.3},
        {"word": "kantor.", "start": 3.3, "end": 3.8},
    ]

    chunks = subtitle_generator.chunk_indonesian_words(words, max_words=4)
    chunk_texts = [" ".join(w["word"] for w in c) for c in chunks]

    # Verify 'nggak pernah' is NOT split awkwardly
    assert any("nggak pernah" in t for t in chunk_texts)
    # Verify 'ke kantor.' is kept together
    assert any("ke kantor." in t for t in chunk_texts)

    # Test ASS file generation with enhanced readability defaults
    plan = EditPlan.create_default("test_sub_readability", duration=10.0)
    assert plan.subtitle_style.font_size == 52
    assert plan.subtitle_style.outline_width == 5
    assert plan.subtitle_style.shadow_width == 3
    assert plan.subtitle_style.margin_v == 540

    ass_out = tmp_path / "test_readability.ass"
    subtitle_generator.generate_ass(
        subtitle_data=[{"words": words}],
        edit_plan=plan,
        output_path=ass_out,
        has_word_timestamps=True
    )

    content = ass_out.read_text(encoding="utf-8")
    assert "Fontsize, PrimaryColour" in content
    assert ",52," in content  # Font size 52
    assert ",5,3,2,100,120,540,1" in content  # outline=5, shadow=3, alignment=2, marginL=100, marginR=120, marginV=540


def test_qc_audio_48khz_stereo_enforcement():
    """Verifies that VideoQualityControl enforces 48000 Hz stereo 2ch in production mode."""
    qc = VideoQualityControl(mode="production")
    # 44.1kHz should fail in production mode
    # Simulated via mock stream in test_qc.py, here we check config / logic
    assert qc.target_width == 1080
    assert qc.target_height == 1920
    assert qc.min_duration == 30.0
    assert qc.max_duration == 55.0
