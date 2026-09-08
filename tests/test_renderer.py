import pytest
from pathlib import Path
from renderer import Renderer
from edit_plan import EditPlan, FramingMode, CropKeyframe, EditEvent, EditEventType


def test_renderer_build_v2_blurred_pipeline():
    ren = Renderer()
    plan = EditPlan.create_default("test_blur", duration=30.0, framing_mode=FramingMode.BLURRED_FALLBACK)
    filter_complex, extra_inputs, map_audio = ren._build_v2_pipeline(
        edit_plan=plan,
        ass_path=None,
        duration=30.0
    )

    # Must contain split for background and foreground
    assert "[bg_in][fg_in]" in filter_complex
    assert "boxblur" in filter_complex
    # Must contain voice-first audio mastering chain by default
    assert "highpass=f=80" in filter_complex
    assert "loudnorm=I=-16.0:TP=-1.5" in filter_complex
    assert map_audio == "[aout]"


def test_renderer_build_v2_face_tracked_pipeline():
    ren = Renderer()
    kfs = [
        CropKeyframe(time=0.0, crop_center_x=0.52, crop_center_y=0.45),
        CropKeyframe(time=2.0, crop_center_x=0.54, crop_center_y=0.45),
    ]
    plan = EditPlan(
        clip_id="test_face",
        clip_duration=25.0,
        framing_mode=FramingMode.FACE_TRACKED,
        crop_keyframes=kfs
    )
    filter_complex, extra_inputs, map_audio = ren._build_v2_pipeline(
        edit_plan=plan,
        ass_path=None,
        duration=25.0
    )

    # Must contain 9:16 portrait crop directly
    assert "crop='ih*9/16':'ih'" in filter_complex
    assert "scale=1080:1920" in filter_complex
    assert "boxblur" not in filter_complex


def test_renderer_build_v2_punch_in_pipeline():
    ren = Renderer()
    plan = EditPlan.create_default("test_punch", duration=20.0)
    plan.edit_events = [
        EditEvent(time=5.0, type=EditEventType.PUNCH_IN, duration=1.5)
    ]
    filter_complex, extra_inputs, map_audio = ren._build_v2_pipeline(
        edit_plan=plan,
        ass_path=None,
        duration=20.0
    )

    # Must contain punch-in conditional crop (939x1669)
    assert "between(t,5.00,6.50)" in filter_complex
    assert "939" in filter_complex
