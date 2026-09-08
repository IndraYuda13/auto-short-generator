import pytest
import json
import subprocess
from pathlib import Path
from qc import VideoQualityControl, QCReport, VideoStreamInfo, AudioStreamInfo


def test_qc_fails_nonexistent_file():
    qc = VideoQualityControl()
    report = qc.evaluate_video("/path/to/definitely_missing_file_12345.mp4")
    assert report.passed is False
    assert "file_exists" in report.checks
    assert report.checks["file_exists"] == "FAIL"
    assert len(report.errors) > 0


def test_qc_fails_zero_byte_file(tmp_path: Path):
    qc = VideoQualityControl()
    dummy_file = tmp_path / "zero_byte.mp4"
    dummy_file.write_bytes(b"")

    report = qc.evaluate_video(str(dummy_file))
    assert report.passed is False
    assert "file_size" in report.checks
    assert "FAIL" in report.checks["file_size"]


def test_qc_duration_modes_semantics(tmp_path: Path, monkeypatch):
    """
    Strict Verification of Duration Semantics:
    - Production QC requires MIN_CLIP_DURATION_SEC=30.0, MAX_CLIP_DURATION_SEC=55.0
    - Fixture QC allows fast smoke tests (e.g. 4.0s)
    - Boundary test matrix:
      * 29.9s -> FAILS production QC, PASSES fixture QC
      * 30.0s -> PASSES production QC
      * 45.0s -> PASSES production QC
      * 55.0s -> PASSES production QC
      * 55.1s -> FAILS production QC
    """
    dummy_file = tmp_path / "test_video.mp4"
    dummy_file.write_bytes(b"dummy_video_payload")

    # Helper to simulate ffprobe json return for arbitrary duration
    def mock_evaluate_duration(qc: VideoQualityControl, dur: float) -> QCReport:
        fake_ffprobe = {
            "streams": [
                {
                    "codec_type": "video",
                    "codec_name": "h264",
                    "width": 1080,
                    "height": 1920,
                    "r_frame_rate": "30/1",
                    "duration": str(dur)
                },
                {
                    "codec_type": "audio",
                    "codec_name": "aac",
                    "channels": 2,
                    "sample_rate": 48000,
                    "duration": str(dur)
                }
            ],
            "format": {
                "duration": str(dur)
            }
        }

        # Monkeypatch subprocess.run to return fake ffprobe stdout
        def fake_run(cmd, *args, **kwargs):
            class Res:
                stdout = json.dumps(fake_ffprobe)
                stderr = ""
                returncode = 0
            return Res()

        monkeypatch.setattr(subprocess, "run", fake_run)
        return qc.evaluate_video(str(dummy_file))

    qc_prod = VideoQualityControl(mode="production")
    qc_fixture = VideoQualityControl(mode="fixture")

    # 1. 29.9s (too short for production Shorts)
    rep_29_9 = mock_evaluate_duration(qc_prod, 29.9)
    assert rep_29_9.passed is False
    assert "below minimum 30.0s" in " ".join(rep_29_9.errors)
    assert rep_29_9.checks["duration_bounds"].startswith("FAIL")

    # 29.9s passes in fixture mode
    rep_29_9_fix = mock_evaluate_duration(qc_fixture, 29.9)
    assert rep_29_9_fix.passed is True
    assert rep_29_9_fix.checks["duration_bounds"].startswith("PASS")

    # 2. 4.0s (smoke test fixture duration)
    rep_4_0 = mock_evaluate_duration(qc_prod, 4.0)
    assert rep_4_0.passed is False
    rep_4_0_fix = mock_evaluate_duration(qc_fixture, 4.0)
    assert rep_4_0_fix.passed is True

    # 3. 30.0s exact lower bound passes production QC
    rep_30_0 = mock_evaluate_duration(qc_prod, 30.0)
    assert rep_30_0.passed is True
    assert rep_30_0.checks["duration_bounds"].startswith("PASS")

    # 4. 45.0s sweet spot passes production QC
    rep_45_0 = mock_evaluate_duration(qc_prod, 45.0)
    assert rep_45_0.passed is True
    assert rep_45_0.checks["duration_bounds"].startswith("PASS")

    # 5. 55.0s exact upper bound passes production QC
    rep_55_0 = mock_evaluate_duration(qc_prod, 55.0)
    assert rep_55_0.passed is True
    assert rep_55_0.checks["duration_bounds"].startswith("PASS")

    # 6. 55.1s exceeds maximum 55.0s for production QC
    rep_55_1 = mock_evaluate_duration(qc_prod, 55.1)
    assert rep_55_1.passed is False
    assert "exceeds maximum 55.0s" in " ".join(rep_55_1.errors)
    assert rep_55_1.checks["duration_bounds"].startswith("FAIL")
