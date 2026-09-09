"""Tests for word_subtitle_engine V3.1 Hybrid Subtitle Accuracy Engine."""

import os
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest

from editing.word_subtitle_engine import (
    generate_hybrid_subtitles,
    generate_ass_from_words,
    validate_ass_timeline,
    chunk_words_to_phrases,
    enforce_non_overlapping_timeline,
)


def test_chunk_words_to_phrases():
    words = [
        {"word": "Dan", "start": 0.0, "end": 0.3},
        {"word": "ketika", "start": 0.3, "end": 0.6},
        {"word": "kita", "start": 0.6, "end": 0.9},
        {"word": "berhasil", "start": 0.9, "end": 1.4},
        {"word": "mencintai", "start": 1.4, "end": 1.8},
        {"word": "diri", "start": 1.8, "end": 2.1},
    ]
    phrases = chunk_words_to_phrases(words, min_words=2, max_words=4)
    assert len(phrases) >= 2
    for p in phrases:
        assert p["start"] < p["end"]
        assert len(p["words"]) <= 4


@patch("transcription.transcript_fusion.extract_word_timestamps_large")
@patch("transcription.transcript_fusion.gemini_verify_transcript")
def test_generate_hybrid_subtitles_mocked(mock_gemini, mock_asr, tmp_path):
    mock_asr.return_value = [
        {"word": "Peren", "start": 12.90, "end": 13.20, "probability": 0.6},
        {"word": "pertama,", "start": 13.20, "end": 13.56, "probability": 0.8},
        {"word": "huayu", "start": 14.54, "end": 15.10, "probability": 0.5},
        {"word": "siapa", "start": 15.10, "end": 15.60, "probability": 0.95},
        {"word": "kamu?", "start": 15.60, "end": 16.62, "probability": 0.95},
    ]

    mock_gemini.return_value = {
        "corrected_full_text": "Pertama, who are you? Siapa kamu?",
        "segments": [
            {
                "asr_text": "Peren pertama, huayu siapa kamu?",
                "corrected_text": "Pertama, who are you? Siapa kamu?",
                "confidence": 0.98,
            }
        ],
        "overall_confidence": 0.98,
        "mode": "GEMINI_NATIVE_VIDEO",
    }

    dummy_video = str(tmp_path / "dummy.mp4")
    Path(dummy_video).write_bytes(b"dummy")
    output_ass = str(tmp_path / "hybrid.ass")

    ok, report, phrases = generate_hybrid_subtitles(
        video_path=dummy_video,
        output_ass_path=output_ass,
        source_transcript_excerpt="Pertama, who are you? Siapa kamu?",
        video_title="Podcast Bilal",
        channel_title="SUARA BERKELAS",
    )

    assert ok is True
    assert report["valid"] is True
    assert report["overlapping_events"] == 0
    assert Path(output_ass).exists()

    content = Path(output_ass).read_text()
    clean_text = content.replace("\\N", " ")
    assert "who" in clean_text and "are you?" in clean_text
    assert "Pertama," in clean_text
    assert "huayu" not in clean_text


@patch("transcription.transcript_fusion.extract_word_timestamps_large")
@patch("transcription.transcript_fusion.gemini_verify_transcript")
def test_generate_ass_from_words_hybrid_delegation(mock_gemini, mock_asr, tmp_path):
    mock_asr.return_value = [
        {"word": "muka", "start": 10.0, "end": 10.5, "probability": 0.98},
        {"word": "bumi", "start": 10.5, "end": 11.0, "probability": 0.98},
    ]
    mock_gemini.return_value = {
        "corrected_full_text": "muka bumi",
        "segments": [],
        "overall_confidence": 0.99,
        "mode": "GEMINI_NATIVE_VIDEO",
    }

    dummy_video = str(tmp_path / "dummy.mp4")
    Path(dummy_video).write_bytes(b"dummy")
    output_ass = str(tmp_path / "delegation.ass")

    ok, report = generate_ass_from_words(
        video_path=dummy_video,
        output_ass_path=output_ass,
        use_hybrid=True,
    )

    assert ok is True
    assert report["valid"] is True
    assert report["overlapping_events"] == 0
    assert report["gemini_mode"] == "GEMINI_NATIVE_VIDEO"
