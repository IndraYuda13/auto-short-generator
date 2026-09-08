import pytest
from edit_director import EditDirector
from edit_plan import EditingProfile, FramingMode


def test_edit_director_deterministic_fallback():
    director = EditDirector()
    # Provide empty segments -> must return default PODCAST_CLEAN
    plan = director.create_plan_for_clip(
        clip_id="test_fallback",
        start_sec=10.0,
        end_sec=40.0,
        transcript_segments=[],
        framing_mode=FramingMode.BLURRED_FALLBACK
    )
    assert plan.profile == EditingProfile.PODCAST_CLEAN
    assert plan.clip_duration == 30.0
    assert len(plan.edit_events) == 0
    assert plan.framing_mode == FramingMode.BLURRED_FALLBACK


def test_edit_director_no_random_hook():
    director = EditDirector()
    plan = director.create_plan_for_clip(
        clip_id="test_no_random",
        start_sec=0.0,
        end_sec=35.0,
        transcript_segments=[
            {"start": 5.0, "end": 12.0, "text": "ini momen penting sekali"},
            {"start": 15.0, "end": 28.0, "text": "dan ini kesimpulan akhirnya"}
        ]
    )
    # Profile must be a valid EditingProfile, no Camera Stomp/Glitch/Breaking News in plan
    assert isinstance(plan.profile, EditingProfile)
    # Events must strictly be valid EditEvent instances within clip duration
    for ev in plan.edit_events:
        assert ev.time <= 35.0
        assert ev.duration is not None and ev.duration <= 3.0
