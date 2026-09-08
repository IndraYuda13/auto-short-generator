from subtitle import subtitle_generator
from edit_plan import EditPlan, SubtitleStyle
from pathlib import Path

def test_subtitle_quality_guard_filters_gibberish():
    """
    Verifies that subtitle quality guard filters censorship bleep artifacts
    and gibberish pseudo-tokens from final ASS subtitles.
    """
    plan = EditPlan(
        clip_id="test_gibberish",
        clip_duration=10.0,
        subtitle_style=SubtitleStyle()
    )

    bad_words = [
        {"word": "Lu", "start": 1.0, "end": 1.4, "probability": 0.9},
        {"word": "sebutin", "start": 1.4, "end": 1.8, "probability": 0.9},
        {"word": "masuk", "start": 1.8, "end": 2.2, "probability": 0.9},
        {"word": "ke", "start": 2.2, "end": 2.5, "probability": 0.9},
        {"word": "segmen", "start": 2.5, "end": 2.9, "probability": 0.9},
        {"word": "kontut", "start": 3.0, "end": 3.5, "probability": 0.3},
        {"word": "bip", "start": 3.5, "end": 3.8, "probability": 0.2},
        {"word": "mem", "start": 3.8, "end": 4.1, "probability": 0.2},
        {"word": "tut", "start": 4.1, "end": 4.5, "probability": 0.2},
    ]

    out_ass = Path("/tmp/test_gibberish_filter.ass")
    subtitle_generator.generate_ass(
        subtitle_data=[{"words": bad_words}],
        edit_plan=plan,
        output_path=out_ass,
        has_word_timestamps=True
    )

    content = out_ass.read_text(encoding="utf-8")
    assert "kontut" not in content.lower()
    assert "bip" not in content.lower()
    assert "mem" not in content.lower()
    assert "tut" not in content.lower()
    assert "Lu sebutin masuk" in content or "sebutin" in content
