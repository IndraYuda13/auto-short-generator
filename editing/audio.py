"""Audio Mastering Module (Blueprint Bab 14).

Enforces:
1. Output mastering: AAC, 48 kHz, stereo, target -16 LUFS, true peak <= -1.5 dB.
2. Filter chain FFmpeg:
   highpass=f=80,loudnorm=I=-16:TP=-1.5:LRA=11,aformat=sample_rates=48000:channel_layouts=stereo
3. Zero SFX! (No sound effects, whooshes, dings, or intrusive background music).
"""

import os
import json
import logging
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)

# Standard broadcast mastering filter chain for vertical speech shorts (Blueprint Bab 14)
AUDIO_FILTER_CHAIN: str = (
    "highpass=f=80,loudnorm=I=-16:TP=-1.5:LRA=11,aformat=sample_rates=48000:channel_layouts=stereo"
)

TARGET_LUFS: float = -16.0
TARGET_TRUE_PEAK: float = -1.5
TARGET_LRA: float = 11.0
TARGET_SAMPLE_RATE: int = 48000
TARGET_CHANNELS: int = 2


def get_audio_filter_chain() -> str:
    """Returns the canonical FFmpeg audio filter chain string for speech mastering."""
    return AUDIO_FILTER_CHAIN


def build_audio_encoding_args() -> List[str]:
    """Returns FFmpeg argument list for encoding standardized AAC 48kHz stereo audio."""
    return [
        "-c:a", "aac",
        "-b:a", "192k",
        "-ar", str(TARGET_SAMPLE_RATE),
        "-ac", str(TARGET_CHANNELS)
    ]


class AudioMasterer:
    """Broadcast audio mastering processor for speech short-form content."""

    def __init__(self):
        self.filter_chain = AUDIO_FILTER_CHAIN

    def master_audio(
        self,
        input_media_path: str,
        output_audio_path: str,
        start_sec: Optional[float] = None,
        end_sec: Optional[float] = None
    ) -> bool:
        """Applies highpass, EBU R128 loudness normalization, and 48kHz stereo formatting.

        Zero SFX policy is strictly maintained.
        """
        if not os.path.exists(input_media_path):
            logger.error(f"Input file not found: {input_media_path}")
            return False

        cmd = ["ffmpeg", "-y"]

        if start_sec is not None and start_sec > 0:
            cmd.extend(["-ss", f"{start_sec:.3f}"])
        if end_sec is not None and (start_sec is None or end_sec > start_sec):
            duration = end_sec - (start_sec or 0.0)
            cmd.extend(["-t", f"{duration:.3f}"])

        cmd.extend([
            "-i", input_media_path,
            "-vn",
            "-af", self.filter_chain,
            "-c:a", "aac",
            "-b:a", "192k",
            "-ar", "48000",
            "-ac", "2",
            output_audio_path
        ])

        try:
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            if res.returncode != 0:
                logger.error(f"Audio mastering failed: {res.stderr}")
                return False
            return os.path.exists(output_audio_path)
        except Exception as e:
            logger.error(f"Audio mastering exception: {e}")
            return False

    def verify_audio_specs(self, media_path: str) -> Dict[str, Any]:
        """Inspects audio stream via ffprobe to verify sample rate, channels, and codec."""
        if not os.path.exists(media_path):
            return {"error": "file_not_found"}

        cmd = [
            "ffprobe",
            "-v", "error",
            "-select_streams", "a:0",
            "-show_entries", "stream=codec_name,sample_rate,channels,channel_layout",
            "-of", "json",
            media_path
        ]
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            if res.returncode == 0:
                data = json.loads(res.stdout or "{}")
                streams = data.get("streams", [])
                if streams:
                    s = streams[0]
                    return {
                        "codec": s.get("codec_name"),
                        "sample_rate": int(s.get("sample_rate", 0)),
                        "channels": int(s.get("channels", 0)),
                        "channel_layout": s.get("channel_layout")
                    }
        except Exception as e:
            logger.debug(f"ffprobe audio verification error: {e}")

        return {"error": "no_audio_stream"}
