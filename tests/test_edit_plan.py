import pytest
from edit_plan import (
    EditPlan,
    EditingProfile,
    FramingMode,
    EditEvent,
    EditEventType,
    CropKeyframe,
    SubtitleStyle,
    AudioProfile,
)


def test_edit_plan_default():
    plan = EditPlan.create_default(clip_id="test_01", duration=45.0)
    assert plan.clip_id == "test_01"
    assert plan.clip_duration == 45.0
    assert plan.profile == EditingProfile.PODCAST_CLEAN
    assert plan.framing_mode == FramingMode.BLURRED_FALLBACK
    assert len(plan.edit_events) == 0
    assert len(plan.crop_keyframes) == 0
    assert plan.subtitle_style.active_word_scale == 100
    assert plan.audio_profile.loudness_target_lufs == -16.0


def test_edit_plan_event_clamping():
    events = [
        EditEvent(time=10.0, type=EditEventType.PUNCH_IN, duration=2.0),
        EditEvent(time=48.0, type=EditEventType.PUNCH_IN, duration=5.0),  # extends past 50.0s
        EditEvent(time=55.0, type=EditEventType.PUNCH_IN, duration=1.0),  # beyond duration, must be discarded
    ]
    plan = EditPlan(
        clip_id="test_clamp",
        clip_duration=50.0,
        profile=EditingProfile.PODCAST_CLEAN,
        framing_mode=FramingMode.FACE_TRACKED,
        edit_events=events
    )

    assert len(plan.edit_events) == 2
    assert plan.edit_events[0].time == 10.0
    assert plan.edit_events[1].time == 48.0
    # Clamped duration: 50.0 - 48.0 = 2.0
    assert plan.edit_events[1].duration == 2.0


def test_crop_keyframe_sorting():
    keyframes = [
        CropKeyframe(time=5.0, crop_center_x=0.5, crop_center_y=0.5),
        CropKeyframe(time=1.0, crop_center_x=0.6, crop_center_y=0.4),
        CropKeyframe(time=3.0, crop_center_x=0.55, crop_center_y=0.45),
    ]
    plan = EditPlan(
        clip_id="test_kf",
        clip_duration=30.0,
        crop_keyframes=keyframes
    )
    times = [k.time for k in plan.crop_keyframes]
    assert times == [1.0, 3.0, 5.0]
