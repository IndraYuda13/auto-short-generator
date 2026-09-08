"""Technical Quality Control (QC) module for rendered video shorts.

Implements Blueprint Bab 16.1:
- Uses ffprobe to inspect rendered video output
- Validates file exists and non-empty (> 100KB)
- Validates video codec == 'h264'
- Validates resolution == 1080x1920
- Validates audio codec == 'aac'
- Validates audio sample_rate == 48000 (48kHz)
- Validates duration within [30.0s, 55.0s] (+/- 0.5s tolerance, i.e. 29.5s - 55.5s)
- Validates no stream corruption via ffmpeg null muxer (return exit 0)
- Exports typed Pydantic TechnicalQCResult data contract
"""

import json
import logging
import os
import subprocess
from pathlib import Path
from typing import List, Optional, Tuple, Union
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class TechnicalQCResult(BaseModel):
    """Structured result of Technical Quality Control evaluation (Blueprint Bab 16.1)."""
    passed: bool = Field(..., description="True if all technical criteria pass without error")
    duration: float = Field(default=0.0, description="Measured duration of the clip in seconds")
    width: int = Field(default=0, description="Video width in pixels")
    height: int = Field(default=0, description="Video height in pixels")
    video_codec: str = Field(default="", description="Video codec name (e.g., 'h264')")
    audio_codec: str = Field(default="", description="Audio codec name (e.g., 'aac')")
    sample_rate: int = Field(default=0, description="Audio sample rate in Hz (e.g., 48000)")
    errors: List[str] = Field(default_factory=list, description="List of detected technical defect descriptions")


class TechnicalQC:
    """Evaluates technical specs and stream integrity of rendered short-form videos."""

    DEFAULT_MIN_DURATION: float = 30.0
    DEFAULT_MAX_DURATION: float = 55.0
    DEFAULT_TOLERANCE: float = 0.5
    DEFAULT_MIN_FILE_SIZE_BYTES: int = 100 * 1024  # 100 KB

    def __init__(
        self,
        min_duration: float = DEFAULT_MIN_DURATION,
        max_duration: float = DEFAULT_MAX_DURATION,
        duration_tolerance: float = DEFAULT_TOLERANCE,
        min_file_size_bytes: int = DEFAULT_MIN_FILE_SIZE_BYTES,
        target_width: int = 1080,
        target_height: int = 1920,
        target_video_codec: str = "h264",
        target_audio_codec: str = "aac",
        target_sample_rate: int = 48000,
        check_stream_corruption: bool = True,
        stream_check_timeout_sec: int = 30,
    ):
        self.min_duration = float(min_duration)
        self.max_duration = float(max_duration)
        self.duration_tolerance = float(duration_tolerance)
        self.min_file_size_bytes = int(min_file_size_bytes)
        self.target_width = int(target_width)
        self.target_height = int(target_height)
        self.target_video_codec = target_video_codec.lower()
        self.target_audio_codec = target_audio_codec.lower()
        self.target_sample_rate = int(target_sample_rate)
        self.check_stream_corruption = check_stream_corruption
        self.stream_check_timeout_sec = stream_check_timeout_sec

    def evaluate(self, video_path: Union[str, Path]) -> TechnicalQCResult:
        """Runs comprehensive ffprobe technical checks and ffmpeg corruption validation."""
        path = Path(video_path)
        errors: List[str] = []

        # 1. Existence and non-empty (> 100KB) check
        if not path.exists():
            return TechnicalQCResult(
                passed=False,
                duration=0.0,
                width=0,
                height=0,
                video_codec="",
                audio_codec="",
                sample_rate=0,
                errors=[f"File does not exist: {path}"]
            )

        try:
            file_size = path.stat().st_size
        except OSError as e:
            return TechnicalQCResult(
                passed=False,
                duration=0.0,
                width=0,
                height=0,
                video_codec="",
                audio_codec="",
                sample_rate=0,
                errors=[f"Cannot access file stat: {e}"]
            )

        if file_size <= self.min_file_size_bytes:
            errors.append(
                f"File size {file_size} bytes is <= {self.min_file_size_bytes} bytes "
                f"({self.min_file_size_bytes / 1024:.1f} KB required)"
            )

        # 2. ffprobe metadata extraction
        probe_cmd = [
            "ffprobe",
            "-v", "error",
            "-print_format", "json",
            "-show_format",
            "-show_streams",
            str(path),
        ]

        try:
            res = subprocess.run(
                probe_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True,
                timeout=20,
            )
            meta = json.loads(res.stdout)
        except subprocess.TimeoutExpired:
            errors.append("ffprobe timed out while reading video metadata")
            return TechnicalQCResult(
                passed=False,
                duration=0.0,
                width=0,
                height=0,
                video_codec="",
                audio_codec="",
                sample_rate=0,
                errors=errors,
            )
        except Exception as e:
            errors.append(f"ffprobe failed to inspect file: {e}")
            return TechnicalQCResult(
                passed=False,
                duration=0.0,
                width=0,
                height=0,
                video_codec="",
                audio_codec="",
                sample_rate=0,
                errors=errors,
            )

        streams = meta.get("streams", [])
        v_stream = next((s for s in streams if s.get("codec_type") == "video"), None)
        a_stream = next((s for s in streams if s.get("codec_type") == "audio"), None)
        format_info = meta.get("format", {})

        # 3. Video stream validation
        v_codec = ""
        width = 0
        height = 0
        v_duration = 0.0

        if not v_stream:
            errors.append("Missing video stream")
        else:
            raw_v_codec = str(v_stream.get("codec_name", "")).lower()
            width = int(v_stream.get("width", 0))
            height = int(v_stream.get("height", 0))

            # Codec check: accepts h264 / avc1
            if raw_v_codec in ("h264", "avc1") or "h264" in raw_v_codec or "avc" in raw_v_codec:
                v_codec = "h264"
            else:
                v_codec = raw_v_codec
                errors.append(
                    f"Invalid video codec '{raw_v_codec}', expected '{self.target_video_codec}'"
                )

            # Resolution check: exactly 1080x1920
            if width != self.target_width or height != self.target_height:
                errors.append(
                    f"Invalid resolution {width}x{height}, expected {self.target_width}x{self.target_height}"
                )

            try:
                v_duration = float(v_stream.get("duration", 0.0) or 0.0)
            except (ValueError, TypeError):
                v_duration = 0.0

        # 4. Audio stream validation
        a_codec = ""
        sample_rate = 0
        a_duration = 0.0

        if not a_stream:
            errors.append("Missing audio stream")
        else:
            raw_a_codec = str(a_stream.get("codec_name", "")).lower()
            sample_rate = int(a_stream.get("sample_rate", 0))

            if "aac" in raw_a_codec:
                a_codec = "aac"
            else:
                a_codec = raw_a_codec
                errors.append(
                    f"Invalid audio codec '{raw_a_codec}', expected '{self.target_audio_codec}'"
                )

            if sample_rate != self.target_sample_rate:
                errors.append(
                    f"Invalid audio sample rate {sample_rate} Hz, expected {self.target_sample_rate} Hz"
                )

            try:
                a_duration = float(a_stream.get("duration", 0.0) or 0.0)
            except (ValueError, TypeError):
                a_duration = 0.0

        # 5. Duration calculation & range check [30.0s, 55.0s] (+/- 0.5s)
        try:
            format_duration = float(format_info.get("duration", 0.0) or 0.0)
        except (ValueError, TypeError):
            format_duration = 0.0

        # Use primary format duration or fallback to video/audio stream duration
        duration = round(format_duration or v_duration or a_duration, 3)

        eff_min = self.min_duration - self.duration_tolerance
        eff_max = self.max_duration + self.duration_tolerance

        if duration < (eff_min - 0.01):
            errors.append(
                f"Duration {duration:.2f}s is below minimum allowed {eff_min:.1f}s "
                f"(target {self.min_duration:.1f}s +/- {self.duration_tolerance:.1f}s)"
            )
        elif duration > (eff_max + 0.01):
            errors.append(
                f"Duration {duration:.2f}s exceeds maximum allowed {eff_max:.1f}s "
                f"(target {self.max_duration:.1f}s +/- {self.duration_tolerance:.1f}s)"
            )

        # 6. Stream corruption validation via ffmpeg null muxer
        if self.check_stream_corruption and path.exists() and file_size > self.min_file_size_bytes:
            is_valid, corruption_err = self._validate_stream_integrity(path)
            if not is_valid:
                errors.append(f"Stream corruption detected: {corruption_err}")

        passed = (len(errors) == 0)

        return TechnicalQCResult(
            passed=passed,
            duration=duration,
            width=width,
            height=height,
            video_codec=v_codec,
            audio_codec=a_codec,
            sample_rate=sample_rate,
            errors=errors,
        )

    def _validate_stream_integrity(self, path: Path) -> Tuple[bool, Optional[str]]:
        """Executes ffmpeg -v error -i <file> -f null - to verify no corruption/decode errors."""
        cmd = [
            "ffmpeg",
            "-v", "error",
            "-xerror",
            "-i", str(path),
            "-f", "null",
            "-",
        ]
        try:
            res = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=self.stream_check_timeout_sec,
            )
            if res.returncode != 0:
                err_text = res.stderr.strip() or f"ffmpeg exited with non-zero code {res.returncode}"
                return False, err_text
            # If returncode is 0, check if any critical error was logged in stderr
            if res.stderr and "error" in res.stderr.lower():
                return False, res.stderr.strip()
            return True, None
        except subprocess.TimeoutExpired:
            return False, f"Integrity check timed out after {self.stream_check_timeout_sec}s"
        except Exception as e:
            return False, f"Failed to execute stream integrity check: {e}"


def evaluate_technical_qc(
    video_path: Union[str, Path],
    min_duration: float = TechnicalQC.DEFAULT_MIN_DURATION,
    max_duration: float = TechnicalQC.DEFAULT_MAX_DURATION,
    duration_tolerance: float = TechnicalQC.DEFAULT_TOLERANCE,
    check_stream_corruption: bool = True,
) -> TechnicalQCResult:
    """Convenience function to evaluate technical QC on a video file."""
    qc = TechnicalQC(
        min_duration=min_duration,
        max_duration=max_duration,
        duration_tolerance=duration_tolerance,
        check_stream_corruption=check_stream_corruption,
    )
    return qc.evaluate(video_path)
