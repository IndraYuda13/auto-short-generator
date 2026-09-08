import pytest
from pathlib import Path
from subtitle import SubtitleGeneratorV2, escape_ass_text, format_ass_time
from edit_plan import EditPlan, SubtitleStyle


def test_escape_ass_text():
    raw = "Kalimat {dengan} kurung \\ backslash"
    escaped = escape_ass_text(raw)
    assert "\\{" in escaped
    assert "\\}" in escaped
    assert "\\\\" in escaped


def test_format_ass_time():
    assert format_ass_time(0.0) == "0:00:00.00"
    assert format_ass_time(65.25) == "0:01:05.25"
    assert format_ass_time(3661.5) == "1:01:01.50"


def test_subtitle_youtube_segment_fallback(tmp_path: Path):
    gen = SubtitleGeneratorV2()
    plan = EditPlan.create_default("clip_yt_fallback", duration=20.0)

    # Segments without word timestamps (YouTube transcript API fallback)
    segments = [
        {"start": 1.0, "duration": 4.0, "end": 5.0, "text": "Ini adalah transkrip resmi YouTube tanpa per kata"},
        {"start": 6.0, "duration": 3.0, "end": 9.0, "text": "Segmen kedua tetap tenang tanpa karaoke palsu"}
    ]
    out_ass = tmp_path / "test_yt.ass"
    gen.generate_ass(segments, plan, out_ass, has_word_timestamps=False)

    content = out_ass.read_text()
    assert "PlayResX: 1080" in content
    assert "PlayResY: 1920" in content
    assert "Dialogue: 0,0:00:01.00,0:00:05.00,Default,,0,0,0,,Ini adalah transkrip" in content
    # Ensure no fabricated per-word karaoke tags (\k or \kf) are present
    assert "\\k" not in content


def test_subtitle_segment_fallback_resolves_overlaps(tmp_path: Path):
    """
    Verifies that overlapping segments (e.g. YouTube API returning 0.0-5.52 and 1.68-6.16)
    are deterministically clamped so that no two dialogue lines overlap in time.
    """
    gen = SubtitleGeneratorV2()
    plan = EditPlan.create_default("clip_overlap", duration=10.0)

    overlapping_segments = [
        {"start": 0.0, "duration": 5.52, "end": 5.52, "text": "Baris dialog pertama"},
        {"start": 1.68, "duration": 4.48, "end": 6.16, "text": "Baris dialog kedua"},
        {"start": 5.52, "duration": 3.00, "end": 8.52, "text": "Baris dialog ketiga"}
    ]
    out_ass = tmp_path / "test_overlap.ass"
    gen.generate_ass(overlapping_segments, plan, out_ass, has_word_timestamps=False)

    lines = [line for line in out_ass.read_text().splitlines() if line.startswith("Dialogue:")]
    assert len(lines) == 3
    # First line should be clamped to end at 1.68s (when second line starts)
    assert "Dialogue: 0,0:00:00.00,0:00:01.68,Default,,0,0,0,,Baris dialog pertama" in lines[0]
    # Second line starts at 1.68s and ends at 5.52s (when third line starts)
    assert "Dialogue: 0,0:00:01.68,0:00:05.52,Default,,0,0,0,,Baris dialog kedua" in lines[1]
    # Third line starts at 5.52s and ends at 8.52s
    assert "Dialogue: 0,0:00:05.52,0:00:08.52,Default,,0,0,0,,Baris dialog ketiga" in lines[2]


def test_subtitle_calm_word_highlighting(tmp_path: Path):
    gen = SubtitleGeneratorV2()
    plan = EditPlan.create_default("clip_words", duration=15.0)
    plan.emphasis_words = ["penting"]

    segments = [
        {
            "start": 0.5,
            "end": 3.0,
            "words": [
                {"word": "ini", "start": 0.5, "end": 1.0},
                {"word": "momen", "start": 1.0, "end": 1.5},
                {"word": "penting", "start": 1.5, "end": 2.2},
                {"word": "banget", "start": 2.2, "end": 2.8},
            ]
        }
    ]
    out_ass = tmp_path / "test_words.ass"
    gen.generate_ass(segments, plan, out_ass, has_word_timestamps=True)

    content = out_ass.read_text()
    assert "Dialogue:" in content
    # The emphasis word 'penting' must have the highlight color tag
    assert "{\\c&H0000E6FF}penting{\\c&H00FFFFFF}" in content
    # Normal words stay calm (no 108% scale pop)
    assert "\\fscx108" not in content
