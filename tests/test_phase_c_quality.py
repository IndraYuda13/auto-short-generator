"""Comprehensive Unit & Integration Test Suite for Phase C (Three-Tier Quality Control Gate).

Implements and verifies Blueprint Bab 16:
1. Bab 16.1: Technical QC (ffprobe validation, codecs, resolution, sample rate, duration tolerances, stream corruption)
2. Bab 16.2: Visual QC (frame sampling, blank/flash detection, subject presence, face framing, subtitle overlap & safe-zones)
3. Bab 16.3: Perceptual QC (3x3 contact sheet, Gemini Visual Director via 9router, blocking issues rejection, max 1 repair)
4. Consolidated ThreeTierQCGate pipeline
"""

import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from quality import (
    TechnicalQC,
    TechnicalQCResult,
    evaluate_technical_qc,
    VisualQC,
    VisualQCResult,
    evaluate_visual_qc,
    PerceptualQC,
    PerceptualQCResult,
    evaluate_perceptual_qc,
    ThreeTierQCGate,
    ThreeTierQCReport,
    evaluate_three_tier_qc,
)

SAMPLE_VIDEO_A = "/root/projects/auto-short-generator-v3/downloads/sample_a_h264_clip.mp4"


# ==============================================================================
# Helpers: Synthetic Media Generators
# ==============================================================================

def create_synthetic_video(
    output_path: str,
    duration_sec: float = 30.0,
    width: int = 1080,
    height: int = 1920,
    video_codec: str = "libx264",
    audio_codec: str = "aac",
    sample_rate: int = 48000,
    bitrate: str = "1500k",
    draw_face: bool = False,
    draw_subtitle_bottom_danger: bool = False,
    draw_subtitle_safe: bool = False,
) -> str:
    """Creates a valid synthetic test MP4 video using ffmpeg."""
    # Video source filter with central foreground subject
    v_filter = f"color=c=black:size={width}x{height}:rate=30,drawbox=x=340:y=500:w=400:h=700:color=red@1.0:t=fill"
    if draw_subtitle_bottom_danger:
        # Draw high-contrast text in bottom 20% danger area (y=1650 on 1920p)
        v_filter += f",drawbox=x=100:y=1650:w=880:h=60:color=white@1.0:t=fill"
    elif draw_subtitle_safe:
        # Draw high-contrast text in safe zone (y=1200 on 1920p)
        v_filter += f",drawbox=x=100:y=1200:w=880:h=60:color=white@1.0:t=fill"

    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", v_filter,
        "-f", "lavfi", "-i", f"sine=frequency=1000:sample_rate={sample_rate}",
        "-t", f"{duration_sec:.2f}",
        "-c:v", video_codec,
        "-preset", "ultrafast",
        "-threads", "4",
        "-b:v", bitrate,
        "-pix_fmt", "yuv420p",
        "-c:a", audio_codec,
        "-ar", str(sample_rate),
        output_path,
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    return output_path


# ==============================================================================
# 1. Bab 16.1: Technical QC Tests
# ==============================================================================

def test_technical_qc_fails_nonexistent_file():
    """Technical QC must fail gracefully for missing file."""
    qc = TechnicalQC()
    result = qc.evaluate("/path/to/missing_video_file_9999.mp4")

    assert result.passed is False
    assert result.duration == 0.0
    assert any("does not exist" in err for err in result.errors)


def test_technical_qc_fails_small_or_empty_file(tmp_path: Path):
    """Technical QC must fail for files <= 100KB."""
    qc = TechnicalQC(min_file_size_bytes=100 * 1024)

    # 1. Zero byte file
    empty_file = tmp_path / "zero_bytes.mp4"
    empty_file.write_bytes(b"")
    r_empty = qc.evaluate(empty_file)
    assert r_empty.passed is False
    assert any("<= 102400 bytes" in err for err in r_empty.errors)

    # 2. 50KB dummy file (< 100KB)
    small_file = tmp_path / "small_50kb.mp4"
    small_file.write_bytes(b"x" * (50 * 1024))
    r_small = qc.evaluate(small_file)
    assert r_small.passed is False
    assert any("<= 102400 bytes" in err for err in r_small.errors)


def test_technical_qc_validates_h264_1080x1920_aac_48khz_pass(tmp_path: Path):
    """Technical QC must pass a video meeting all Blueprint 16.1 technical criteria."""
    video_path = str(tmp_path / "valid_30s.mp4")
    # Generate 30.0s 1080x1920 H.264 AAC 48kHz with > 100KB size
    create_synthetic_video(video_path, duration_sec=30.0, bitrate="1500k")

    assert os.path.getsize(video_path) > 100 * 1024

    qc = TechnicalQC()
    result = qc.evaluate(video_path)

    assert result.passed is True
    assert result.width == 1080
    assert result.height == 1920
    assert result.video_codec == "h264"
    assert result.audio_codec == "aac"
    assert result.sample_rate == 48000
    assert 29.5 <= result.duration <= 55.5
    assert len(result.errors) == 0


def test_technical_qc_duration_boundaries(tmp_path: Path, monkeypatch):
    """Verifies duration validation [30.0s, 55.0s] with +/- 0.5s tolerance:

    - < 29.5s: FAILS
    - 29.5s: PASSES
    - 45.0s: PASSES
    - 55.5s: PASSES
    - > 55.5s: FAILS
    """
    dummy_file = tmp_path / "dummy_video.mp4"
    dummy_file.write_bytes(b"x" * (120 * 1024))  # > 100KB

    def mock_evaluate_duration(dur: float) -> TechnicalQCResult:
        fake_ffprobe = {
            "streams": [
                {
                    "codec_type": "video",
                    "codec_name": "h264",
                    "width": 1080,
                    "height": 1920,
                    "duration": str(dur),
                },
                {
                    "codec_type": "audio",
                    "codec_name": "aac",
                    "sample_rate": 48000,
                    "duration": str(dur),
                },
            ],
            "format": {
                "duration": str(dur),
            },
        }

        def fake_run(cmd, *args, **kwargs):
            class Res:
                stdout = json.dumps(fake_ffprobe)
                stderr = ""
                returncode = 0
            return Res()

        monkeypatch.setattr(subprocess, "run", fake_run)
        qc = TechnicalQC(check_stream_corruption=False)
        return qc.evaluate(str(dummy_file))

    # 1. 29.4s: Below minimum 29.5s (30.0 - 0.5)
    r_29_4 = mock_evaluate_duration(29.4)
    assert r_29_4.passed is False
    assert any("below minimum allowed 29.5s" in err for err in r_29_4.errors)

    # 2. 29.5s: Exact lower bound passes
    r_29_5 = mock_evaluate_duration(29.5)
    assert r_29_5.passed is True
    assert len(r_29_5.errors) == 0

    # 3. 40.0s: Mid-range passes
    r_40_0 = mock_evaluate_duration(40.0)
    assert r_40_0.passed is True

    # 4. 55.5s: Exact upper bound passes (55.0 + 0.5)
    r_55_5 = mock_evaluate_duration(55.5)
    assert r_55_5.passed is True

    # 5. 55.6s: Exceeds maximum 55.5s
    r_55_6 = mock_evaluate_duration(55.6)
    assert r_55_6.passed is False
    assert any("exceeds maximum allowed 55.5s" in err for err in r_55_6.errors)


def test_technical_qc_codec_and_resolution_violations(tmp_path: Path, monkeypatch):
    """Verifies that invalid codecs, resolutions, or sample rates fail with clear errors."""
    dummy_file = tmp_path / "dummy_video.mp4"
    dummy_file.write_bytes(b"x" * (120 * 1024))

    def mock_eval(width=1080, height=1920, v_codec="h264", a_codec="aac", sample_rate=48000, has_audio=True):
        streams = [
            {
                "codec_type": "video",
                "codec_name": v_codec,
                "width": width,
                "height": height,
                "duration": "40.0",
            }
        ]
        if has_audio:
            streams.append({
                "codec_type": "audio",
                "codec_name": a_codec,
                "sample_rate": sample_rate,
                "duration": "40.0",
            })

        fake_ffprobe = {
            "streams": streams,
            "format": {"duration": "40.0"},
        }

        def fake_run(cmd, *args, **kwargs):
            class Res:
                stdout = json.dumps(fake_ffprobe)
                stderr = ""
                returncode = 0
            return Res()

        monkeypatch.setattr(subprocess, "run", fake_run)
        qc = TechnicalQC(check_stream_corruption=False)
        return qc.evaluate(str(dummy_file))

    # Invalid resolution 1920x1080 (horizontal landscape)
    r_res = mock_eval(width=1920, height=1080)
    assert r_res.passed is False
    assert any("Invalid resolution 1920x1080" in err for err in r_res.errors)

    # Invalid video codec (vp9)
    r_vcode = mock_eval(v_codec="vp9")
    assert r_vcode.passed is False
    assert any("Invalid video codec 'vp9'" in err for err in r_vcode.errors)

    # Invalid audio codec (mp3)
    r_acode = mock_eval(a_codec="mp3")
    assert r_acode.passed is False
    assert any("Invalid audio codec 'mp3'" in err for err in r_acode.errors)

    # Invalid sample rate (44100 Hz instead of 48000 Hz)
    r_sr = mock_eval(sample_rate=44100)
    assert r_sr.passed is False
    assert any("Invalid audio sample rate 44100 Hz" in err for err in r_sr.errors)

    # Missing audio stream entirely
    r_no_audio = mock_eval(has_audio=False)
    assert r_no_audio.passed is False
    assert any("Missing audio stream" in err for err in r_no_audio.errors)


def test_technical_qc_stream_corruption_detection(tmp_path: Path):
    """Technical QC detects corrupted video streams via ffmpeg null muxer check."""
    corrupt_file = tmp_path / "corrupt_stream.mp4"
    # Write garbage header/payload that cannot decode
    corrupt_file.write_bytes(b"ftypmp42" + os.urandom(150 * 1024))

    qc = TechnicalQC(check_stream_corruption=True)
    result = qc.evaluate(corrupt_file)

    assert result.passed is False
    assert len(result.errors) > 0


# ==============================================================================
# 2. Bab 16.2: Visual QC Tests
# ==============================================================================

def test_visual_qc_detects_blank_black_and_white_frames():
    """Visual QC detects blank/black (mean < 5) and white blown-out (mean > 250) frames."""
    qc = VisualQC(min_frames=5)

    # Create synthetic frames: 5 normal, 1 pure black, 1 pure white
    h, w = 1920, 1080
    normal_frame = np.full((h, w, 3), 120, dtype=np.uint8)
    black_frame = np.zeros((h, w, 3), dtype=np.uint8)        # mean = 0 < 5
    white_frame = np.full((h, w, 3), 255, dtype=np.uint8)    # mean = 255 > 250

    frames = [
        (1.0, normal_frame),
        (2.0, normal_frame),
        (3.0, black_frame),
        (4.0, normal_frame),
        (5.0, white_frame),
    ]

    blank_count, errors = qc.check_blank_and_flashes(frames)
    assert blank_count == 2
    assert any("Detected 2 blank/black/white frames" in err for err in errors)


def test_visual_qc_detects_black_and_white_flashes():
    """Visual QC detects isolated 1-frame black or white flashes between normal frames."""
    qc = VisualQC()
    h, w = 480, 270
    normal_1 = np.full((h, w, 3), 110, dtype=np.uint8)
    flash_black = np.zeros((h, w, 3), dtype=np.uint8)  # Isolated black frame
    normal_2 = np.full((h, w, 3), 115, dtype=np.uint8)

    frames = [
        (1.0, normal_1),
        (2.0, flash_black),
        (3.0, normal_2),
    ]

    blank_count, errors = qc.check_blank_and_flashes(frames)
    assert any("Detected black/white flash at 2.0s" in err for err in errors)


def test_visual_qc_subject_presence_ratio():
    """Visual QC flags videos where subject is missing for excessive frames (> 30%)."""
    qc = VisualQC(min_subject_ratio=0.70)
    h, w = 1920, 1080

    # Clean frame with central subject contour
    subject_frame = np.full((h, w, 3), 50, dtype=np.uint8)
    cv2.circle(subject_frame, (w // 2, h // 2), 300, (220, 220, 220), -1)

    # Empty flat background frame (no subject)
    empty_frame = np.full((h, w, 3), 100, dtype=np.uint8)

    # 4 frames with subject, 6 empty frames -> 40% presence ratio (< 70% threshold)
    sampled = [(float(i), subject_frame if i < 4 else empty_frame) for i in range(10)]

    ratio, errors = qc.check_subject_and_face_framing(sampled)
    assert ratio < 0.70
    assert any("Subject missing in excessive frames" in err for err in errors)


def test_visual_qc_face_cut_badly_top_headroom(monkeypatch):
    """Visual QC flags face cut off at top canvas boundary without proper headroom."""
    qc = VisualQC()
    h, w = 1920, 1080
    frame = np.full((h, w, 3), 100, dtype=np.uint8)

    # Mock face detector to return a face touching the top border (y=0)
    monkeypatch.setattr(qc, "detect_faces", lambda f: [(400, 0, 280, 300, 0.95)])

    sampled = [(1.0, frame), (2.0, frame)]
    ratio, errors = qc.check_subject_and_face_framing(sampled)

    assert any("Face cut badly at top (no headroom)" in err for err in errors)


def test_visual_qc_face_cut_badly_bottom_edge(monkeypatch):
    """Visual QC flags face truncated at bottom canvas boundary."""
    qc = VisualQC()
    h, w = 1920, 1080
    frame = np.full((h, w, 3), 100, dtype=np.uint8)

    # Mock face detector to return face truncated at bottom: y=1650, h=270 on H=1920 (y+h = 1920)
    monkeypatch.setattr(qc, "detect_faces", lambda f: [(400, 1650, 280, 270, 0.95)])

    sampled = [(1.0, frame), (2.0, frame)]
    ratio, errors = qc.check_subject_and_face_framing(sampled)

    assert any("Face cut badly at bottom edge" in err for err in errors)


def test_visual_qc_subtitle_duplicate_stuck_detection():
    """Visual QC flags stuck subtitles persisting identically across > 4 consecutive sampled frames."""
    qc = VisualQC()
    h, w = 1920, 1080

    # Create frame with subtitle in safe zone
    frame_with_sub = np.full((h, w, 3), 80, dtype=np.uint8)
    # Draw high contrast subtitle text in lower third
    cv2.putText(
        frame_with_sub, "TEKS SUBTITLE YANG MACET TERUS MENERUS",
        (100, 1400), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 4
    )

    # 6 consecutive sampled frames with identical subtitle
    sampled = [(float(i * 2), frame_with_sub.copy()) for i in range(6)]

    errors = qc.check_subtitle_overlap_and_duplicates(sampled)
    assert any("Duplicate or stuck subtitle detected" in err for err in errors)


def test_visual_qc_boundary_safe_zone_violations():
    """Visual QC detects subtitle safe-zone violations in top 15% or bottom 20% danger areas."""
    qc = VisualQC()
    h, w = 1920, 1080

    # 1. Subtitle violating top 15% danger zone (y < 288px on 1920p)
    top_violating_frame = np.full((h, w, 3), 50, dtype=np.uint8)
    cv2.putText(
        top_violating_frame, "SUBTITLE DI AREA BAHAYA ATAS 15%",
        (100, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 4
    )
    sampled_top = [(float(i), top_violating_frame.copy()) for i in range(3)]
    safe_top, errors_top = qc.check_boundary_safe_zone(sampled_top)

    assert safe_top is False
    assert any("top 15% danger area" in err for err in errors_top)

    # 2. Subtitle violating bottom 20% danger zone (y > 1536px on 1920p)
    bottom_violating_frame = np.full((h, w, 3), 50, dtype=np.uint8)
    cv2.putText(
        bottom_violating_frame, "SUBTITLE DI AREA BAHAYA BAWAH 20%",
        (100, 1750), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 4
    )
    sampled_bottom = [(float(i), bottom_violating_frame.copy()) for i in range(3)]
    safe_bottom, errors_bottom = qc.check_boundary_safe_zone(sampled_bottom)

    assert safe_bottom is False
    assert any("bottom 20% danger area" in err for err in errors_bottom)

    # 3. Subtitle in safe zone (y = 1350px on 1920p, between 288 and 1536)
    clean_frame = np.full((h, w, 3), 50, dtype=np.uint8)
    cv2.putText(
        clean_frame, "SUBTITLE DI SAFE ZONE 1350PX",
        (100, 1350), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 4
    )
    sampled_clean = [(float(i), clean_frame.copy()) for i in range(3)]
    safe_clean, errors_clean = qc.check_boundary_safe_zone(sampled_clean)

    assert safe_clean is True
    assert len(errors_clean) == 0


def test_visual_qc_real_sample_video_pass():
    """Runs Visual QC on real media clip (sample_a_h264_clip.mp4)."""
    if not os.path.exists(SAMPLE_VIDEO_A):
        pytest.skip(f"Sample video not found at {SAMPLE_VIDEO_A}")

    qc = VisualQC(sample_interval_sec=2.0, min_frames=5)
    result = qc.evaluate(SAMPLE_VIDEO_A)

    assert result.sampled_frames_count >= 5
    assert result.blank_frames == 0
    assert result.subject_present_ratio > 0.50
    assert result.subtitle_safe is True


# ==============================================================================
# 3. Bab 16.3: Perceptual QC Tests (Gemini Visual Director via 9router)
# ==============================================================================

def test_perceptual_qc_generates_3x3_contact_sheet(tmp_path: Path):
    """Perceptual QC must generate a 3x3 contact sheet image containing 9 frames."""
    video_path = str(tmp_path / "contact_sheet_test.mp4")
    create_synthetic_video(video_path, duration_sec=5.0, bitrate="1000k")

    qc = PerceptualQC()
    sheet_img, b64_str = qc.generate_contact_sheet(
        video_path,
        grid_size=(3, 3),
        cell_size=(240, 426),
    )

    assert sheet_img is not None
    assert b64_str is not None
    # 3 rows of 426 height = 1278, 3 cols of 240 width = 720
    assert sheet_img.shape[0] == 3 * 426
    assert sheet_img.shape[1] == 3 * 240
    assert len(b64_str) > 1000


def test_perceptual_qc_approval_when_clean(monkeypatch, tmp_path: Path):
    """When Gemini returns publishable=True, score >= 70, and no blocking issues: PASS."""
    video_path = str(tmp_path / "clean_clip.mp4")
    create_synthetic_video(video_path, duration_sec=5.0)

    qc = PerceptualQC()

    fake_response = {
        "publishable": True,
        "score": 88,
        "blocking_issues": [],
        "notes": "Framing subjek sangat rapi, subtitle kontras tinggi di safe zone, transisi bersih."
    }

    class FakeResp:
        def raise_for_status(self): pass
        def json(self):
            return {
                "choices": [
                    {"message": {"content": json.dumps(fake_response)}}
                ]
            }

    monkeypatch.setattr("requests.post", lambda *args, **kwargs: FakeResp())

    res = qc.evaluate(
        video_path=video_path,
        transcript_text="Podcast seru hari ini",
        technical_qc={"passed": True},
        visual_qc={"passed": True},
    )

    assert res.passed is True
    assert res.publishable is True
    assert res.score == 88
    assert len(res.blocking_issues) == 0
    assert "Framing subjek" in res.notes


def test_perceptual_qc_rejection_when_blocking_issues(monkeypatch, tmp_path: Path):
    """If blocking_issues are present: strictly reject/skip (publishable=False, passed=False)."""
    video_path = str(tmp_path / "defective_clip.mp4")
    create_synthetic_video(video_path, duration_sec=5.0)

    qc = PerceptualQC()

    fake_response = {
        "publishable": False,
        "score": 45,
        "blocking_issues": ["Wajah pembicara terpotong batas canvas atas", "Subtitle menabrak UI bawah"],
        "notes": "Video tidak layak publish karena ada pelanggaran framing fatal."
    }

    class FakeResp:
        def raise_for_status(self): pass
        def json(self):
            return {
                "choices": [
                    {"message": {"content": json.dumps(fake_response)}}
                ]
            }

    monkeypatch.setattr("requests.post", lambda *args, **kwargs: FakeResp())

    res = qc.evaluate(
        video_path=video_path,
        transcript_text="Podcast rusak",
    )

    assert res.passed is False
    assert res.publishable is False
    assert res.score == 45
    assert len(res.blocking_issues) == 2
    assert "Wajah pembicara terpotong" in res.blocking_issues[0]


def test_perceptual_qc_deterministic_repair_max_one_attempt(monkeypatch, tmp_path: Path):
    """Perceptual QC attempts maximum 1 deterministic repair if blocking issues exist."""
    video_path = str(tmp_path / "repair_clip.mp4")
    repaired_path = str(tmp_path / "repaired_clip.mp4")
    create_synthetic_video(video_path, duration_sec=5.0)
    create_synthetic_video(repaired_path, duration_sec=5.0)

    qc = PerceptualQC()

    eval_calls = 0

    def fake_post(*args, **kwargs):
        nonlocal eval_calls
        eval_calls += 1
        class FakeResp:
            def raise_for_status(self): pass
            def json(self):
                if eval_calls == 1:
                    # First run: blocking issue detected
                    return {
                        "choices": [
                            {"message": {"content": json.dumps({
                                "publishable": False,
                                "score": 50,
                                "blocking_issues": ["Headroom cut badly"],
                                "notes": "Perlu geser crop window ke bawah 50px."
                            })}}
                        ]
                    }
                else:
                    # Second run on repaired clip: clean pass
                    return {
                        "choices": [
                            {"message": {"content": json.dumps({
                                "publishable": True,
                                "score": 90,
                                "blocking_issues": [],
                                "notes": "Setelah perbaikan headroom, framing sempurna."
                            })}}
                        ]
                    }
        return FakeResp()

    monkeypatch.setattr("requests.post", fake_post)

    repair_invocations = 0

    def mock_repair_handler(issues: List[str], plan: Any) -> Tuple[bool, Optional[str], Optional[str]]:
        nonlocal repair_invocations
        repair_invocations += 1
        return True, repaired_path, "Shifted crop window downward by 50px"

    res = qc.evaluate(
        video_path=video_path,
        repair_handler=mock_repair_handler,
    )

    # Verify repair was invoked exactly once
    assert repair_invocations == 1
    assert eval_calls == 2
    assert res.passed is True
    assert res.publishable is True
    assert res.repair_attempted is True
    assert res.repair_action == "Shifted crop window downward by 50px"


def test_perceptual_qc_deterministic_fallback_on_offline_router(monkeypatch, tmp_path: Path):
    """Perceptual QC falls back deterministically when 9router is unreachable or times out."""
    video_path = str(tmp_path / "fallback_clip.mp4")
    create_synthetic_video(video_path, duration_sec=5.0)

    qc = PerceptualQC()

    # Simulate network connection error
    def mock_post_raise(*args, **kwargs):
        raise ConnectionError("Failed to establish connection to 9router on port 20128")

    monkeypatch.setattr("requests.post", mock_post_raise)

    # 1. Technical & Visual QC passed -> Fallback approves
    res_pass = qc.evaluate(
        video_path=video_path,
        technical_qc={"passed": True, "errors": []},
        visual_qc={"passed": True, "errors": []},
    )
    assert res_pass.passed is True
    assert res_pass.publishable is True
    assert "Deterministic fallback" in res_pass.notes

    # 2. Technical QC failed -> Fallback rejects
    res_fail = qc.evaluate(
        video_path=video_path,
        technical_qc={"passed": False, "errors": ["Audio sample rate invalid"]},
        visual_qc={"passed": True, "errors": []},
    )
    assert res_fail.passed is False
    assert res_fail.publishable is False
    assert any("Audio sample rate invalid" in err for err in res_fail.blocking_issues)


# ==============================================================================
# 4. Consolidated ThreeTierQCGate Tests
# ==============================================================================

def test_three_tier_qc_gate_end_to_end_orchestration(tmp_path: Path, monkeypatch):
    """ThreeTierQCGate runs Tier 1, Tier 2, and Tier 3 in unified sequence."""
    video_path = str(tmp_path / "threetier_clip.mp4")
    create_synthetic_video(video_path, duration_sec=30.0, bitrate="1500k")

    gate = ThreeTierQCGate()

    # Mock perceptual QC response
    fake_perc_resp = {
        "publishable": True,
        "score": 92,
        "blocking_issues": [],
        "notes": "Kualitas visual sangat baik di semua tier."
    }

    class FakeResp:
        def raise_for_status(self): pass
        def json(self):
            return {
                "choices": [
                    {"message": {"content": json.dumps(fake_perc_resp)}}
                ]
            }

    monkeypatch.setattr("requests.post", lambda *args, **kwargs: FakeResp())

    report = gate.evaluate(
        video_path=video_path,
        transcript_text="Transkrip lengkap untuk tiga lapis QC.",
    )

    assert isinstance(report, ThreeTierQCReport)
    assert report.technical.passed is True
    assert report.visual.passed is True
    assert report.perceptual.passed is True
    assert report.passed is True
    assert report.publishable is True
    assert len(report.errors) == 0
