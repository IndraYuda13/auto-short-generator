import pytest
from pathlib import Path
from renderer import Renderer
from edit_plan import EditPlan, FramingMode, CropKeyframe, EditEvent, EditEventType, AudioProfile
from config import settings


def test_renderer_matrix_subtitles_on_vs_off(tmp_path: Path):
    """
    Matrix Test: Subtitles ON vs Subtitles OFF.
    - Subtitles ON: contains ass filter with font configuration.
    - Subtitles OFF: no ass filter, passes through clean base video.
    """
    ren = Renderer()
    plan = EditPlan.create_default("test_sub", duration=20.0)

    # Subtitles OFF (ass_path=None)
    filter_off, extra_off, _ = ren._build_v2_pipeline(plan, ass_path=None, duration=20.0)
    assert "ass='" not in filter_off
    assert "null[outv]" in filter_off

    # Subtitles ON (dummy ass file)
    dummy_ass = tmp_path / "test.ass"
    dummy_ass.write_text("[Script Info]\nTitle: Test", encoding="utf-8")
    filter_on, extra_on, _ = ren._build_v2_pipeline(plan, ass_path=dummy_ass, duration=20.0)
    assert "ass='" in filter_on
    assert "fontsdir=" in filter_on
    assert "[outv]" in filter_on


def test_renderer_matrix_audio_mastering_on_vs_off(monkeypatch):
    """
    Matrix Test: Audio Mastering ON vs Audio Mastering OFF.
    - Audio Mastering ON: applies voice-first mastering filter chain (highpass 80Hz, acompressor, loudnorm -16 LUFS)
    - Audio Mastering OFF: passes original 0:a stream directly without modification
    """
    ren = Renderer()
    plan = EditPlan.create_default("test_audio", duration=20.0)

    # 1. Mastering ON (settings.AUDIO_MASTERING_ENABLED = True)
    monkeypatch.setattr(settings, "AUDIO_MASTERING_ENABLED", True)
    filter_on, _, map_audio_on = ren._build_v2_pipeline(plan, ass_path=None, duration=20.0)
    assert map_audio_on == "[aout]"
    assert "highpass=f=80" in filter_on
    assert "acompressor" in filter_on
    assert "loudnorm=I=-16.0:TP=-1.5" in filter_on

    # 2. Mastering OFF (settings.AUDIO_MASTERING_ENABLED = False)
    monkeypatch.setattr(settings, "AUDIO_MASTERING_ENABLED", False)
    filter_off, _, map_audio_off = ren._build_v2_pipeline(plan, ass_path=None, duration=20.0)
    assert map_audio_off == "0:a"
    assert "loudnorm" not in filter_off
    assert "highpass" not in filter_off


def test_renderer_matrix_framing_face_tracked_vs_blurred_fallback():
    """
    Matrix Test: Face-tracked framing vs Blurred fallback.
    - Face-tracked: direct 9:16 portrait crop centered on speaker, no boxblur background.
    - Blurred fallback: split into bg and fg, boxblur 5:2 applied to bg, fg centered over bg.
    """
    ren = Renderer()

    # Face-tracked
    kfs = [
        CropKeyframe(time=0.0, crop_center_x=0.55, crop_center_y=0.45),
        CropKeyframe(time=1.0, crop_center_x=0.55, crop_center_y=0.45)
    ]
    plan_face = EditPlan(
        clip_id="test_face",
        clip_duration=30.0,
        framing_mode=FramingMode.FACE_TRACKED,
        crop_keyframes=kfs
    )
    filter_face, _, _ = ren._build_v2_pipeline(plan_face, ass_path=None, duration=30.0)
    assert "crop='ih*9/16':'ih'" in filter_face
    assert "boxblur" not in filter_face

    # Blurred fallback
    plan_blur = EditPlan.create_default("test_blur", duration=30.0, framing_mode=FramingMode.BLURRED_FALLBACK)
    filter_blur, _, _ = ren._build_v2_pipeline(plan_blur, ass_path=None, duration=30.0)
    assert "boxblur=5:2" in filter_blur
    assert "split=2[bg_in][fg_in]" in filter_blur


def test_renderer_matrix_punch_in_requested_by_edit_plan():
    """
    Matrix Test: Punch-in event requested by EditPlan.
    - When requested: conditional 115% zoom crop generated strictly for specified event window.
    - When none: no zoom crop expression present.
    """
    ren = Renderer()

    # Without punch-in
    plan_no_punch = EditPlan.create_default("no_punch", duration=15.0)
    filter_no_punch, _, _ = ren._build_v2_pipeline(plan_no_punch, ass_path=None, duration=15.0)
    assert "punch_v" not in filter_no_punch

    # With punch-in at t=4.0s for 1.5s
    plan_punch = EditPlan.create_default("punch", duration=15.0)
    plan_punch.edit_events = [
        EditEvent(time=4.0, type=EditEventType.PUNCH_IN, duration=1.5, intensity=1.15)
    ]
    filter_punch, _, _ = ren._build_v2_pipeline(plan_punch, ass_path=None, duration=15.0)
    assert "punch_v" in filter_punch
    assert "between(t,4.00,5.50)" in filter_punch
    assert "939" in filter_punch
    assert "1669" in filter_punch


def test_renderer_matrix_no_semantic_or_random_effects_invented():
    """
    Strict Requirement: Verify NO random or unrequested visual effects are invented by Renderer V2.
    - No random hue, camera shake, strobe, glitch, or random overlays when executing EditPlan.
    """
    ren = Renderer()
    plan = EditPlan.create_default("clean_deterministic", duration=25.0)
    filter_complex, _, _ = ren._build_v2_pipeline(plan, ass_path=None, duration=25.0)

    forbidden_tokens = [
        "glitch", "camera_stomp", "paper_tear", "breaking_news",
        "random", "hue", "noise", "drawbox", "frei0r"
    ]
    for token in forbidden_tokens:
        assert token not in filter_complex.lower(), f"Forbidden effect '{token}' found in V2 pipeline filtergraph"
