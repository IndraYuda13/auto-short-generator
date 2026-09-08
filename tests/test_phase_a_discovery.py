"""Unit tests for Phase A Discovery package (searcher and source_filter)."""

import pytest
from discovery.searcher import Searcher, VideoMetadata, VideoSourceMeta
from discovery.source_filter import (
    SourceFilter,
    SourceFilterVerdict,
    EligibilityResult,
)


def test_video_metadata_model():
    """Verify VideoMetadata and VideoSourceMeta fields and defaults."""
    meta = VideoMetadata(
        video_id="abc123xyz",
        url="https://www.youtube.com/watch?v=abc123xyz",
        title="Podcast Obrolan Bisnis",
        duration=600.0,
        view_count=50000,
        channel="Bisnis Channel",
    )
    assert meta.video_id == "abc123xyz"
    assert meta.duration == 600.0
    assert meta.duration_sec == 600.0
    assert meta.is_live is False
    assert meta.is_upcoming is False
    assert meta.is_private is False

    # Also verify VideoSourceMeta instantiation with duration_sec
    v_source = VideoSourceMeta(
        video_id="xyz789",
        url="https://www.youtube.com/watch?v=xyz789",
        title="Interview Kreator",
        channel="Channel Kreatif",
        duration_sec=1200.0,
        published_at="2024-05-01",
        view_count=10000,
        description="Deskripsi interview",
    )
    assert v_source.video_id == "xyz789"
    assert v_source.duration_sec == 1200.0
    assert v_source.duration == 1200.0


def test_source_filter_accepts_valid_longform_video():
    """Valid Indonesian longform video is accepted."""
    filter_engine = SourceFilter(min_duration_sec=180.0)
    meta = VideoMetadata(
        video_id="valid_vid_01",
        url="https://www.youtube.com/watch?v=valid_vid_01",
        title="Curhat Bang Denny Sumargo bareng Bintang Tamu Spesial",
        duration=3600.0,
        view_count=120000,
        channel="Curhat Bang",
    )
    verdict = filter_engine.filter_video(meta)
    assert verdict.accepted is True
    assert verdict.is_eligible is True
    assert verdict.rejection_code is None
    assert "satisfies Phase A" in verdict.reason


def test_source_filter_rejects_duration_too_short():
    """Video shorter than 180 seconds is rejected early."""
    filter_engine = SourceFilter(min_duration_sec=180.0)
    meta = VideoMetadata(
        video_id="short_vid_02",
        url="https://www.youtube.com/watch?v=short_vid_02",
        title="Cuplikan 1 Menit",
        duration=65.0,
        view_count=1000,
    )
    verdict = filter_engine.filter_video(meta)
    assert verdict.accepted is False
    assert verdict.is_eligible is False
    assert verdict.rejection_code == "DURATION_TOO_SHORT"
    assert "below minimum 180s" in verdict.reason


def test_source_filter_rejects_duration_too_long():
    """Video longer than 14400 seconds (4 hours) is rejected."""
    filter_engine = SourceFilter(max_duration_sec=14400.0)
    meta = VideoMetadata(
        video_id="long_vid_02b",
        url="https://www.youtube.com/watch?v=long_vid_02b",
        title="Marathon Livestream 5 Jam",
        duration=18000.0,
    )
    verdict = filter_engine.filter_video(meta)
    assert verdict.accepted is False
    assert verdict.is_eligible is False
    assert verdict.rejection_code == "DURATION_TOO_LONG"
    assert "14400s" in verdict.reason


def test_source_filter_rejects_live_stream():
    """Live streams are rejected."""
    filter_engine = SourceFilter()
    meta = VideoMetadata(
        video_id="live_vid_03",
        url="https://www.youtube.com/watch?v=live_vid_03",
        title="Live Streaming Podcast",
        duration=7200.0,
        is_live=True,
    )
    verdict = filter_engine.filter_video(meta)
    assert verdict.accepted is False
    assert verdict.is_eligible is False
    assert verdict.rejection_code == "IS_LIVE"


def test_source_filter_rejects_upcoming_premiere():
    """Upcoming streams or premieres are rejected."""
    filter_engine = SourceFilter()
    meta = VideoMetadata(
        video_id="upcoming_vid_03b",
        url="https://www.youtube.com/watch?v=upcoming_vid_03b",
        title="Premiere Podcast Malam Ini",
        duration=3600.0,
        is_upcoming=True,
    )
    verdict = filter_engine.filter_video(meta)
    assert verdict.accepted is False
    assert verdict.is_eligible is False
    assert verdict.rejection_code == "IS_UPCOMING"


def test_source_filter_rejects_duplicate_video():
    """Duplicate videos in processed list are rejected."""
    filter_engine = SourceFilter(processed_video_ids={"already_done_id"})
    meta = VideoMetadata(
        video_id="already_done_id",
        url="https://www.youtube.com/watch?v=already_done_id",
        title="Podcast Keren",
        duration=1200.0,
    )
    verdict = filter_engine.filter_video(meta)
    assert verdict.accepted is False
    assert verdict.rejection_code == "DUPLICATE_VIDEO"


def test_source_filter_rejects_music_only_video():
    """Music-only or non-speech pattern in title is rejected."""
    filter_engine = SourceFilter()
    meta = VideoMetadata(
        video_id="music_vid_04",
        url="https://www.youtube.com/watch?v=music_vid_04",
        title="Official Music Video Band Pop Indonesia - Kenangan",
        duration=240.0,
    )
    verdict = filter_engine.filter_video(meta)
    assert verdict.accepted is False
    assert verdict.rejection_code == "NO_CLEAR_SPEECH"


def test_source_filter_rejects_full_movie_and_asmr_and_no_commentary():
    """Full movie, ASMR, and gameplay no commentary are rejected."""
    filter_engine = SourceFilter()

    # Full movie
    v_movie = VideoMetadata(
        video_id="movie_01",
        url="https://www.youtube.com/watch?v=movie_01",
        title="Film Bioskop Full Movie HD",
        duration=5400.0,
    )
    assert filter_engine.filter_video(v_movie).rejection_code == "NO_CLEAR_SPEECH"

    # ASMR
    v_asmr = VideoMetadata(
        video_id="asmr_01",
        url="https://www.youtube.com/watch?v=asmr_01",
        title="Relaxing Mukbang ASMR Crunchy Food",
        duration=1200.0,
    )
    assert filter_engine.filter_video(v_asmr).rejection_code == "NO_CLEAR_SPEECH"

    # Gameplay no commentary
    v_gameplay = VideoMetadata(
        video_id="game_01",
        url="https://www.youtube.com/watch?v=game_01",
        title="Elden Ring Gameplay Walkthrough No Commentary Part 1",
        duration=3600.0,
    )
    assert filter_engine.filter_video(v_gameplay).rejection_code == "NO_CLEAR_SPEECH"


def test_source_filter_batch_processing():
    """Batch filtering marks accepted videos and tracks duplicates."""
    filter_engine = SourceFilter(min_duration_sec=180.0)
    videos = [
        VideoMetadata(video_id="v1", url="url1", title="Podcast Satu", duration=500.0),
        VideoMetadata(video_id="v2", url="url2", title="Short Video", duration=60.0),
        VideoMetadata(video_id="v1", url="url1", title="Podcast Satu Duplikat", duration=500.0),
    ]
    verdicts = filter_engine.filter_batch(videos)
    assert len(verdicts) == 3
    assert verdicts[0].accepted is True
    assert verdicts[1].accepted is False  # too short
    assert verdicts[2].accepted is False  # duplicate v1


def test_searcher_direct_filter_connection():
    """Searcher connects directly with SourceFilter so only eligible candidates pass."""
    filter_engine = SourceFilter()
    searcher = Searcher(source_filter=filter_engine)

    candidates = [
        VideoSourceMeta(video_id="c1", url="u1", title="Too Short", duration_sec=50.0),
        VideoSourceMeta(video_id="c2", url="u2", title="Music Video Official MV", duration_sec=240.0),
        VideoSourceMeta(video_id="c3", url="u3", title="Valid Podcast Ngobrol Santai", duration_sec=1800.0),
    ]

    # Mock low-level search to return candidates without downloading
    searcher._search_via_ytdlp = lambda q, max_results=5: candidates

    filtered = searcher.search_videos("podcast santai", max_results=5, filter_eligible=True)
    assert len(filtered) == 1
    assert filtered[0].video_id == "c3"
    assert filtered[0].title == "Valid Podcast Ngobrol Santai"
