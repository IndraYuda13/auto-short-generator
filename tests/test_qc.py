import pytest
from pathlib import Path
from qc import VideoQualityControl


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
