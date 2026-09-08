import os
import json
import logging
import subprocess
import shutil
from enum import Enum
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
import cv2
import numpy as np
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class SubtitleSource(str, Enum):
    NONE = "NONE"
    EMBEDDED_TRACK = "EMBEDDED_TRACK"
    BURNED_IN = "BURNED_IN"


class SubtitleRegion(BaseModel):
    x1: float = Field(..., ge=0.0, le=1.0, description="Normalized left X [0.0, 1.0]")
    y1: float = Field(..., ge=0.0, le=1.0, description="Normalized top Y [0.0, 1.0]")
    x2: float = Field(..., ge=0.0, le=1.0, description="Normalized right X [0.0, 1.0]")
    y2: float = Field(..., ge=0.0, le=1.0, description="Normalized bottom Y [0.0, 1.0]")
    protected: bool = Field(default=True, description="Whether this subtitle region must be protected from crop truncation")


class SubtitleDetectionResult(BaseModel):
    source: SubtitleSource = SubtitleSource.NONE
    has_existing_subtitle: bool = False
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    region: Optional[SubtitleRegion] = None
    reason: str = ""
    candidate_detected: bool = False


class SubtitleDetector:
    """
    Detects embedded subtitle streams via ffprobe or detects candidate burned-in subtitles
    via fast edge density and morphology in the lower frame region.
    """

    def __init__(self):
        pass

    def detect_embedded_tracks(self, video_path: str) -> List[Dict[str, Any]]:
        """Uses ffprobe to detect existing subtitle streams in media container."""
        cmd = [
            "ffprobe",
            "-v", "error",
            "-select_streams", "s",
            "-show_entries", "stream=index,codec_name:stream_tags=language,title",
            "-print_format", "json",
            str(video_path)
        ]
        try:
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=8)
            if res.returncode == 0:
                data = json.loads(res.stdout)
                return data.get("streams", [])
        except Exception as e:
            logger.warning(f"ffprobe subtitle track detection error: {e}")
        return []

    def detect_burned_in_candidates(
        self,
        video_path: str,
        start_sec: float,
        end_sec: float,
        num_samples: int = 4,
        sub_y_start_ratio: float = 0.70
    ) -> Tuple[bool, float, Optional[SubtitleRegion], str]:
        """
        Fast OpenCV edge and morphological scan across 3-5 frames in lower third.
        Completes in <0.5 seconds without hanging on heavy OCR.
        Returns: (candidate_found, confidence, SubtitleRegion, reason)
        """
        duration = max(0.1, end_sec - start_sec)
        step = duration / float(num_samples + 1)
        sample_times = [start_sec + step * (i + 1) for i in range(num_samples)]

        positive_frames = 0
        min_x = 1.0
        max_x = 0.0

        for st in sample_times:
            cmd = [
                "ffmpeg", "-y", "-ss", f"{st:.2f}", "-i", str(video_path),
                "-vframes", "1", "-f", "image2pipe", "-vcodec", "png", "-"
            ]
            try:
                proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=5)
                if proc.returncode != 0 or not proc.stdout:
                    continue
                arr = np.frombuffer(proc.stdout, np.uint8)
                frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if frame is None:
                    continue
            except Exception as e:
                logger.debug(f"Frame extraction error at {st}s: {e}")
                continue

            src_h, src_w = frame.shape[:2]
            y_start = int(src_h * sub_y_start_ratio)
            y_end = int(src_h * 0.98)
            roi = frame[y_start:y_end, :]

            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            # High-pass filter / Canny
            edges = cv2.Canny(gray, 80, 200)
            density = float(np.sum(edges > 0)) / float(edges.size)

            # Subtitle text generally exhibits structured edge density between 2% and 18%
            if 0.02 <= density <= 0.20:
                # Morphological closing to connect letter strokes into words
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
                closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
                col_proj = np.sum(closed > 0, axis=0)
                active_cols = np.where(col_proj > (roi.shape[0] * 0.10))[0]

                # If horizontal span of text-like clusters exceeds 30% of frame width
                if len(active_cols) > int(src_w * 0.30):
                    positive_frames += 1
                    min_x = min(min_x, float(active_cols[0]) / src_w)
                    max_x = max(max_x, float(active_cols[-1]) / src_w)

        if positive_frames >= 2:
            x1 = max(0.0, round(min_x - 0.05, 3)) if min_x < 1.0 else 0.06
            x2 = min(1.0, round(max_x + 0.05, 3)) if max_x > 0.0 else 0.94
            region = SubtitleRegion(x1=x1, y1=0.70, x2=x2, y2=0.96, protected=True)
            return True, 0.85, region, f"Fast local edge analysis found wide text clusters in {positive_frames}/{num_samples} frames."

        return False, 0.0, None, f"No prominent subtitle text clusters detected locally ({positive_frames}/{num_samples} frames)."

    def evaluate(self, video_path: str, start_sec: float, end_sec: float) -> SubtitleDetectionResult:
        """
        Runs hierarchical subtitle detection:
        1. ffprobe embedded subtitle tracks -> EMBEDDED_TRACK
        2. Fast local visual candidate check -> BURNED_IN candidate
        3. Otherwise -> NONE
        """
        # Step 1: Check embedded tracks
        tracks = self.detect_embedded_tracks(video_path)
        if tracks:
            codec = tracks[0].get("codec_name", "subrip")
            lang = tracks[0].get("tags", {}).get("language", "unknown")
            return SubtitleDetectionResult(
                source=SubtitleSource.EMBEDDED_TRACK,
                has_existing_subtitle=True,
                confidence=1.0,
                region=SubtitleRegion(x1=0.05, y1=0.70, x2=0.95, y2=0.95, protected=True),
                reason=f"Embedded subtitle stream detected (codec={codec}, language={lang}).",
                candidate_detected=True
            )

        # Step 2: Check local burned-in candidate
        cand_found, conf, region, reason = self.detect_burned_in_candidates(
            video_path=video_path,
            start_sec=start_sec,
            end_sec=end_sec
        )
        if cand_found:
            return SubtitleDetectionResult(
                source=SubtitleSource.BURNED_IN,
                has_existing_subtitle=True,
                confidence=conf,
                region=region,
                reason=reason,
                candidate_detected=True
            )

        return SubtitleDetectionResult(
            source=SubtitleSource.NONE,
            has_existing_subtitle=False,
            confidence=0.95,
            region=None,
            reason=reason,
            candidate_detected=False
        )


subtitle_detector = SubtitleDetector()
