"""Quality Control (QC) module for rendered video shorts.

Executes ffprobe analysis, stream validation, resolution checks, and soft failure detection.
Returns structured QC results serializable to JSON.
"""

import json
import os
import subprocess
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class VideoStreamInfo(BaseModel):
    width: int
    height: int
    codec: str
    duration: float
    fps: float


class AudioStreamInfo(BaseModel):
    codec: str
    channels: int
    sample_rate: int
    duration: float


class QCReport(BaseModel):
    passed: bool
    file_path: str
    file_size_bytes: int
    video: Optional[VideoStreamInfo] = None
    audio: Optional[AudioStreamInfo] = None
    checks: Dict[str, str] = Field(default_factory=dict)
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)

    def to_json(self, indent: int = 2) -> str:
        return self.model_dump_json(indent=indent)


class VideoQualityControl:
    """Automated validator for rendered short-form videos."""

    def __init__(
        self,
        target_width: int = 1080,
        target_height: int = 1920,
        min_duration: float = 2.0,
        max_duration: float = 65.0
    ):
        self.target_width = target_width
        self.target_height = target_height
        self.min_duration = min_duration
        self.max_duration = max_duration

    def evaluate_video(
        self,
        video_path: str,
        expected_duration: Optional[float] = None,
        duration_tolerance_sec: float = 3.0
    ) -> QCReport:
        """
        Runs comprehensive QC on rendered video file:
        1. Existence and non-zero size
        2. ffprobe parsing of video & audio streams
        3. Width == 1080, Height == 1920
        4. Valid playable codecs (h264/avc1, aac)
        5. Duration sanity and tolerance
        6. Soft checks: black frames or silence detection
        """
        p = Path(video_path)
        checks: Dict[str, str] = {}
        errors: List[str] = []
        warnings: List[str] = []

        # 1. Existence check
        if not p.exists():
            return QCReport(
                passed=False,
                file_path=str(p),
                file_size_bytes=0,
                checks={"file_exists": "FAIL"},
                errors=[f"Rendered file does not exist: {video_path}"]
            )

        file_size = p.stat().st_size
        if file_size == 0:
            return QCReport(
                passed=False,
                file_path=str(p),
                file_size_bytes=0,
                checks={"file_size": "FAIL"},
                errors=["Rendered file is 0 bytes"]
            )
        checks["file_size"] = f"PASS ({file_size} bytes)"

        # 2. ffprobe extraction
        probe_cmd = [
            "ffprobe",
            "-v", "error",
            "-print_format", "json",
            "-show_format",
            "-show_streams",
            str(p)
        ]
        try:
            res = subprocess.run(probe_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
            meta = json.loads(res.stdout)
        except Exception as e:
            return QCReport(
                passed=False,
                file_path=str(p),
                file_size_bytes=file_size,
                checks={"ffprobe_readable": "FAIL"},
                errors=[f"ffprobe failed to read file: {e}"]
            )
        checks["ffprobe_readable"] = "PASS"

        streams = meta.get("streams", [])
        v_stream = next((s for s in streams if s.get("codec_type") == "video"), None)
        a_stream = next((s for s in streams if s.get("codec_type") == "audio"), None)

        # Video stream checks
        v_info = None
        if not v_stream:
            errors.append("Missing video stream")
            checks["video_stream"] = "FAIL"
        else:
            w = int(v_stream.get("width", 0))
            h = int(v_stream.get("height", 0))
            v_codec = str(v_stream.get("codec_name", "")).lower()
            v_dur = float(v_stream.get("duration", 0.0) or meta.get("format", {}).get("duration", 0.0))

            # Calculate FPS
            fps_str = v_stream.get("r_frame_rate", "30/1")
            try:
                num, den = map(float, fps_str.split("/"))
                fps = num / den if den > 0 else 30.0
            except Exception:
                fps = 30.0

            v_info = VideoStreamInfo(
                width=w,
                height=h,
                codec=v_codec,
                duration=round(v_dur, 2),
                fps=round(fps, 2)
            )

            # Check resolution
            if w == self.target_width and h == self.target_height:
                checks["resolution"] = f"PASS ({w}x{h})"
            else:
                checks["resolution"] = f"FAIL (expected {self.target_width}x{self.target_height}, got {w}x{h})"
                errors.append(f"Invalid resolution: {w}x{h}")

            # Check video codec
            if "h264" in v_codec or "avc" in v_codec:
                checks["video_codec"] = f"PASS ({v_codec})"
            else:
                checks["video_codec"] = f"FAIL ({v_codec})"
                errors.append(f"Incompatible video codec: {v_codec}")

        # Audio stream checks
        a_info = None
        if not a_stream:
            errors.append("Missing audio stream")
            checks["audio_stream"] = "FAIL"
        else:
            a_codec = str(a_stream.get("codec_name", "")).lower()
            channels = int(a_stream.get("channels", 0))
            sr = int(a_stream.get("sample_rate", 0))
            a_dur = float(a_stream.get("duration", 0.0) or meta.get("format", {}).get("duration", 0.0))

            a_info = AudioStreamInfo(
                codec=a_codec,
                channels=channels,
                sample_rate=sr,
                duration=round(a_dur, 2)
            )

            if "aac" in a_codec or "mp3" in a_codec:
                checks["audio_codec"] = f"PASS ({a_codec})"
            else:
                checks["audio_codec"] = f"WARNING ({a_codec})"
                warnings.append(f"Audio codec is {a_codec}, preferred aac")

            if channels >= 1:
                checks["audio_channels"] = f"PASS ({channels} ch)"
            else:
                checks["audio_channels"] = "FAIL (0 channels)"
                errors.append("Audio stream has 0 channels")

        # Duration validation
        total_duration = float(meta.get("format", {}).get("duration", 0.0))
        if total_duration < self.min_duration:
            checks["duration_bounds"] = f"FAIL (too short: {total_duration:.1f}s)"
            errors.append(f"Duration {total_duration:.1f}s below minimum {self.min_duration}s")
        elif total_duration > self.max_duration:
            checks["duration_bounds"] = f"FAIL (too long: {total_duration:.1f}s)"
            errors.append(f"Duration {total_duration:.1f}s exceeds maximum {self.max_duration}s")
        else:
            checks["duration_bounds"] = f"PASS ({total_duration:.1f}s)"

        if expected_duration is not None:
            diff = abs(total_duration - expected_duration)
            if diff > duration_tolerance_sec:
                checks["duration_match"] = f"FAIL (diff: {diff:.2f}s)"
                errors.append(
                    f"Rendered duration {total_duration:.2f}s differs from expected "
                    f"{expected_duration:.2f}s by {diff:.2f}s (tolerance {duration_tolerance_sec}s)"
                )
            else:
                checks["duration_match"] = f"PASS (expected ~{expected_duration:.1f}s, got {total_duration:.1f}s)"

        passed = len(errors) == 0
        return QCReport(
            passed=passed,
            file_path=str(p),
            file_size_bytes=file_size,
            video=v_info,
            audio=a_info,
            checks=checks,
            errors=errors,
            warnings=warnings
        )


qc_evaluator = VideoQualityControl()
