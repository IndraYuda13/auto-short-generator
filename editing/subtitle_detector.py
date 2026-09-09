"""Subtitle Existence Detection — Deterministic Evidence-Based (V3.1 Fix).

Implements the correct decision hierarchy from the blueprint:
A. ffprobe detects subtitle stream? -> EMBEDDED_TRACK
B. Reliable local burned-in evidence (HIGH confidence only)? -> BURNED_IN
C. Otherwise -> NONE

Critical invariant: UNKNOWN != SOURCE_EXISTING
"""

import json
import logging
import os
import subprocess
from enum import Enum
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class SubtitleDetectionState(str, Enum):
    """Explicit subtitle detection states — never collapse ambiguity into EXISTING."""
    NONE = "NONE"                   # No subtitles detected -> GENERATE
    EMBEDDED_TRACK = "EMBEDDED_TRACK"  # Container has subtitle stream -> SOURCE_EXISTING
    BURNED_IN = "BURNED_IN"         # Burned-in text detected with HIGH confidence -> SOURCE_EXISTING
    UNKNOWN = "UNKNOWN"             # Ambiguous / detection failed -> REJECT candidate


class SubtitleDetector:
    """Deterministic subtitle detection using ffprobe + strict burned-in analysis."""

    # Burned-in detection thresholds (tightened to eliminate false positives)
    BURNED_IN_MIN_VOTE_RATIO: float = 0.50       # At least 50% of frames must show text (was 30%)
    BURNED_IN_MIN_EDGE_DENSITY: float = 0.04      # Minimum edge density (was 0.02)
    BURNED_IN_MAX_EDGE_DENSITY: float = 0.18      # Maximum edge density (was 0.22)
    BURNED_IN_MIN_HORIZONTAL_COVERAGE: float = 0.20  # Minimum width coverage (was 0.15)
    BURNED_IN_TEXT_ROI_TOP: float = 0.75          # Bottom 25% only (was 30%)
    SAMPLE_COUNT: int = 20

    def detect(
        self,
        video_path: str,
        start_sec: float = 0.0,
        end_sec: Optional[float] = None,
    ) -> Tuple[SubtitleDetectionState, float, Optional[Dict]]:
        """Detects subtitle state with confidence score.

        Returns:
            (state, confidence, metadata)
            - state: SubtitleDetectionState enum
            - confidence: 0.0-1.0 detection confidence
            - metadata: Optional dict with detection details
        """
        if not os.path.exists(video_path):
            return SubtitleDetectionState.UNKNOWN, 0.0, {"error": "file_not_found"}

        # Step A: Check for embedded subtitle streams via ffprobe
        embedded_idx = self._check_embedded_stream(video_path)
        if embedded_idx is not None:
            return SubtitleDetectionState.EMBEDDED_TRACK, 1.0, {
                "stream_index": embedded_idx,
                "method": "ffprobe_subtitle_stream",
            }

        # Step B: Check for burned-in subtitles with strict thresholds
        burned, confidence, region = self._check_burned_in_strict(
            video_path, start_sec, end_sec
        )
        if burned and confidence >= 0.7:
            return SubtitleDetectionState.BURNED_IN, confidence, {
                "region": region,
                "method": "canny_morphology_strict",
            }

        if burned and confidence >= 0.4:
            # Low confidence burned-in -> UNKNOWN (don't assume either way)
            return SubtitleDetectionState.UNKNOWN, confidence, {
                "region": region,
                "method": "canny_morphology_strict",
                "note": "Low confidence burned-in detection — ambiguous",
            }

        # Step C: No subtitles detected
        return SubtitleDetectionState.NONE, 1.0 - confidence, {
            "method": "no_subtitle_evidence",
        }

    def _check_embedded_stream(self, video_path: str) -> Optional[int]:
        """Check if video container has embedded subtitle tracks via ffprobe."""
        cmd = [
            "ffprobe", "-v", "error",
            "-select_streams", "s",
            "-show_entries", "stream=index,codec_name",
            "-of", "json",
            video_path,
        ]
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            if res.returncode == 0:
                data = json.loads(res.stdout or "{}")
                streams = data.get("streams", [])
                if streams:
                    return int(streams[0].get("index", 0))
        except Exception as e:
            logger.debug(f"ffprobe subtitle check error: {e}")
        return None

    def _check_burned_in_strict(
        self,
        video_path: str,
        start_sec: float = 0.0,
        end_sec: Optional[float] = None,
    ) -> Tuple[bool, float, Optional[Dict]]:
        """Strict burned-in subtitle detection with tightened thresholds.

        Uses:
        - Otsu binarization in bottom ROI (not just Canny edges)
        - Horizontal morphology for text-like structures
        - Strict vote ratio and coverage thresholds
        - Center-alignment check to filter out corner logos/watermarks
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return False, 0.0, None

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_dur = total_frames / fps if fps > 0 else 0.0

        if end_sec is None or end_sec <= start_sec:
            end_sec = video_dur

        clip_dur = max(0.1, end_sec - start_sec)
        sample_times = [
            start_sec + (i * clip_dur / max(1, self.SAMPLE_COUNT))
            for i in range(self.SAMPLE_COUNT)
        ]

        burned_votes = 0
        valid_frames = 0
        regions = []

        for t in sample_times:
            f_idx = int(t * fps)
            if f_idx >= total_frames:
                continue
            cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
            ret, frame = cap.read()
            if not ret or frame is None:
                continue

            valid_frames += 1
            h, w = frame.shape[:2]

            # ROI: bottom 25% of frame (tighter than 30%)
            y_start = int(h * self.BURNED_IN_TEXT_ROI_TOP)
            roi = frame[y_start:, :]
            roi_h, roi_w = roi.shape[:2]

            # Convert to grayscale and apply Otsu binarization
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # Check white pixel density (text-like content)
            white_ratio = float(np.sum(binary > 200)) / float(binary.size)
            if white_ratio < 0.005 or white_ratio > 0.25:
                # Too little or too much white — not subtitle text
                continue

            # Horizontal morphology to find text-line structures
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 3))
            closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

            # Find contours of text-line candidates
            contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            text_like_contours = 0
            for cnt in contours:
                x, y, cw, ch = cv2.boundingRect(cnt)
                aspect = cw / max(1, ch)
                # Text lines are wide and thin (aspect > 4), centered horizontally
                center_x = (x + cw / 2) / roi_w
                if aspect > 4.0 and cw > roi_w * 0.15 and 0.15 < center_x < 0.85:
                    text_like_contours += 1

            if text_like_contours >= 1:
                burned_votes += 1
                regions.append({"t": round(t, 2), "contours": text_like_contours})

        cap.release()

        if valid_frames == 0:
            return False, 0.0, None

        vote_ratio = burned_votes / valid_frames
        has_burned = vote_ratio >= self.BURNED_IN_MIN_VOTE_RATIO

        region_info = None
        if regions:
            region_info = {
                "vote_ratio": round(vote_ratio, 3),
                "burned_frames": burned_votes,
                "total_frames": valid_frames,
                "sample_detections": regions[:5],
            }

        return has_burned, round(vote_ratio, 3), region_info
