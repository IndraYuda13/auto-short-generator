"""Tests for Auto Clipper V3.1 Hybrid Subtitle Accuracy & Fusion Engine."""

import pytest
from pathlib import Path
from transcription.transcript_fusion import (
    build_context_prompt,
    fuse_transcript,
    generate_fused_ass,
    validate_ass_timeline,
    enforce_non_overlapping_timeline,
)
from quality.gemini_video_qc import GeminiVideoQCResult, GeminiNativeVideoQC


def test_build_context_prompt():
    prompt = build_context_prompt(
        video_title="Podcast Bilal: Self Love & Mindset",
        channel_title="SUARA BERKELAS",
    )
    assert "SUARA BERKELAS" in prompt
    assert "Bilal" in prompt
    assert "muka bumi" in prompt
    assert "nomor tiga" in prompt


def test_fuse_transcript_alignment():
    # Simulated Whisper ASR words with phonetic inaccuracies
    asr_words = [
        {"word": "Peren", "start": 12.90, "end": 13.20, "probability": 0.6},
        {"word": "pertama,", "start": 13.20, "end": 13.56, "probability": 0.8},
        {"word": "huayu", "start": 14.54, "end": 15.10, "probability": 0.5},
        {"word": "siapa", "start": 15.10, "end": 15.60, "probability": 0.95},
        {"word": "kamu?", "start": 15.60, "end": 16.62, "probability": 0.95},
    ]

    gemini_result = {
        "corrected_full_text": "Pertama, who are you? Siapa kamu?",
        "segments": [
            {
                "asr_text": "Peren pertama, huayu siapa kamu?",
                "corrected_text": "Pertama, who are you? Siapa kamu?",
                "confidence": 0.98,
            }
        ],
        "overall_confidence": 0.98,
    }

    phrases = fuse_transcript(asr_words, gemini_result)
    assert len(phrases) >= 1
    # Check that text was corrected
    full_fused_text = " ".join(p["text"] for p in phrases)
    assert "who are you?" in full_fused_text
    assert "Pertama," in full_fused_text
    assert "huayu" not in full_fused_text
    assert "Peren" not in full_fused_text

    # Audio timing invariant: spans must come from Whisper timestamps
    assert phrases[0]["start"] >= 12.90
    assert phrases[-1]["end"] <= 16.65
    assert phrases[0]["confidence"] == "HIGH"
    assert "gemini_native_video" in phrases[0]["evidence"]


def test_generate_fused_ass(tmp_path):
    fused_phrases = [
        {
            "text": "Dan ketika kita berhasil",
            "start": 0.0,
            "end": 1.76,
            "confidence": "HIGH",
            "evidence": ["whisper", "gemini_native_video"],
        },
        {
            "text": "mencintai diri kita sendiri",
            "start": 1.78,
            "end": 3.28,
            "confidence": "HIGH",
            "evidence": ["whisper", "gemini_native_video"],
        },
        {
            "text": "di muka bumi ini, Mas.",
            "start": 10.24,
            "end": 12.34,
            "confidence": "HIGH",
            "evidence": ["whisper", "gemini_native_video"],
        },
    ]

    output_ass = str(tmp_path / "test_output.ass")
    ok, report, validated = generate_fused_ass(fused_phrases, output_ass)

    assert ok is True
    assert report["valid"] is True
    assert report["overlapping_events"] == 0
    assert report["max_simultaneous"] <= 1
    assert Path(output_ass).exists()

    content = Path(output_ass).read_text()
    assert "di muka" in content and "bumi ini, Mas." in content
    assert "Montserrat" in content


def test_gemini_video_qc_result_schema():
    # Pass case
    qc_pass = GeminiVideoQCResult(
        passed=True,
        score=88,
        subtitle_timing="PASS",
        subtitle_overlap="PASS",
        subtitle_linger="PASS",
        subtitle_text_accuracy="PASS",
        obvious_transcription_errors=[],
        blocking_reasons=[],
        summary="All good",
    )
    assert qc_pass.passed is True
    assert qc_pass.subtitle_text_accuracy == "PASS"

    # Fail case with obvious transcription error
    qc_fail = GeminiVideoQCResult(
        passed=False,
        score=65,
        subtitle_timing="PASS",
        subtitle_overlap="PASS",
        subtitle_linger="PASS",
        subtitle_text_accuracy="FAIL",
        obvious_transcription_errors=[
            {"shown": "mukabomi", "heard": "muka bumi", "approx_time": "00:11"}
        ],
        blocking_reasons=["Obvious subtitle error: 'mukabomi' instead of heard 'muka bumi'"],
        summary="Rejected due to phonetic hallucination",
    )
    assert qc_fail.passed is False
    assert qc_fail.subtitle_text_accuracy == "FAIL"
    assert len(qc_fail.obvious_transcription_errors) == 1


def test_gemini_video_qc_string_errors_defensive():
    import json
    from unittest.mock import MagicMock
    mock_client = MagicMock()
    # Case 1: Gemini returns ["tidak ada"] string
    mock_client.video_completion.return_value = '{"passed": true, "score": 85, "subtitle_timing": "PASS", "subtitle_overlap": "PASS", "subtitle_linger": "PASS", "subtitle_text_accuracy": "PASS", "obvious_transcription_errors": ["tidak ada"], "blocking_reasons": [], "summary": "Bagus"}'
    mock_client.extract_json.side_effect = lambda t: json.loads(t)

    evaluator = GeminiNativeVideoQC(client=mock_client)
    res = evaluator.evaluate_video(video_path="/root/projects/auto-short-generator-v3/downloads/fQbpsIQpi08.mp4")
    assert res.passed is True
    assert res.subtitle_text_accuracy == "PASS"
    assert res.blocking_reasons == []

    # Case 2: Gemini returns list of real error strings
    mock_client.video_completion.return_value = '{"passed": true, "score": 85, "subtitle_timing": "PASS", "subtitle_overlap": "PASS", "subtitle_linger": "PASS", "subtitle_text_accuracy": "PASS", "obvious_transcription_errors": ["salah kata mukabomi bukannya muka bumi"], "blocking_reasons": [], "summary": "Ada salah"}'
    res2 = evaluator.evaluate_video(video_path="/root/projects/auto-short-generator-v3/downloads/fQbpsIQpi08.mp4")
    assert res2.passed is False
    assert res2.subtitle_text_accuracy == "FAIL"
    assert any("mukabomi" in b for b in res2.blocking_reasons)

