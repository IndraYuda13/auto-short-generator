"""Visual Analyzer module for Auto Short Generator Phase A.

Performs deterministic local computer vision analysis on video clips via OpenCV:
1. Scene cut timestamp detection (frame diff / histogram distance).
2. Blank / black / white frame detection.
3. Face / person presence ratio (using Haar cascade / YuNet / contrast fallback).
4. Burned-in subtitle detection (edge density and morphology in bottom 30%).
"""

import os
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import cv2
import numpy as np
from pydantic import BaseModel, Field

from config import settings

logger = logging.getLogger(__name__)


class VisualAnalysisReport(BaseModel):
    """Deterministic local visual metrics across a video clip window."""
    video_path: str
    start_sec: float
    end_sec: float
    duration: float
    scene_cuts: List[float] = Field(default_factory=list)
    scene_cut_rate_per_sec: float = 0.0
    blank_frame_ratio: float = 0.0
    subject_presence_ratio: float = 1.0
    has_burned_subtitles: bool = False
    burned_subtitle_region: Optional[Dict[str, float]] = None
    is_viable: bool = True
    rejection_reasons: List[str] = Field(default_factory=list)


class VisualAnalyzer:
    """Local OpenCV-based visual inspector for candidate clip windows."""

    # Default thresholds
    MAX_BLANK_FRAME_RATIO: float = 0.10      # Reject if >10% blank frames
    MIN_SUBJECT_PRESENCE_RATIO: float = 0.20 # Lowered for SAFE_WIDE: podcasts have B-roll/slides where face detector fails
    MAX_SCENE_CUT_RATE: float = 1.2          # Reject if >1.2 cuts/sec (too chaotic)

    def __init__(self):
        # Locate Haar cascade model for face detection
        cascade_path = Path("/root/projects/auto-short-generator-v3/assets/models/haarcascade_frontalface_default.xml")
        if not cascade_path.exists():
            cv2_data = getattr(cv2, "data", None)
            if cv2_data and hasattr(cv2_data, "haarcascades"):
                cascade_path = Path(cv2_data.haarcascades) / "haarcascade_frontalface_default.xml"

        self.face_cascade = None
        classifier_cls = getattr(cv2, "CascadeClassifier", None)
        if cascade_path.exists() and classifier_cls is not None:
            self.face_cascade = classifier_cls(str(cascade_path))

    def analyze_clip(
        self,
        video_path: str,
        start_sec: float,
        end_sec: float,
        sample_fps: float = 2.0,
    ) -> VisualAnalysisReport:
        """Convenience alias for analyze_window matching Orchestrator contract."""
        return self.analyze_window(
            video_path=video_path,
            start_sec=start_sec,
            end_sec=end_sec,
            sample_fps=sample_fps,
        )

    def analyze_window(
        self,
        video_path: str,
        start_sec: float,
        end_sec: float,
        sample_fps: float = 2.0,
    ) -> VisualAnalysisReport:
        """
        Performs comprehensive local vision scan across the candidate window [start_sec, end_sec].
        """
        duration = max(0.1, end_sec - start_sec)
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return VisualAnalysisReport(
                video_path=video_path,
                start_sec=start_sec,
                end_sec=end_sec,
                duration=duration,
                is_viable=False,
                rejection_reasons=["Unable to open video stream with OpenCV"]
            )

        video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_duration = 1.0 / video_fps

        start_frame = int(start_sec * video_fps)
        end_frame = min(total_frames, int(end_sec * video_fps))

        # Sample interval in frames
        step_frames = max(1, int(video_fps / sample_fps))
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        current_frame_idx = start_frame
        sampled_frames: List[Tuple[float, np.ndarray]] = []

        while current_frame_idx <= end_frame:
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_idx)
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            timestamp = current_frame_idx * frame_duration
            sampled_frames.append((timestamp, frame))
            current_frame_idx += step_frames

        cap.release()

        if not sampled_frames:
            return VisualAnalysisReport(
                video_path=video_path,
                start_sec=start_sec,
                end_sec=end_sec,
                duration=duration,
                is_viable=False,
                rejection_reasons=["No frames could be extracted from video window"]
            )

        # 1. Blank frame detection
        blank_count = 0
        for ts, frame in sampled_frames:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            mean_val = float(np.mean(gray))
            std_val = float(np.std(gray))
            # Black frame (mean < 12) or white frame (mean > 243) or pure static (std < 5.0)
            if mean_val < 12.0 or mean_val > 243.0 or std_val < 5.0:
                blank_count += 1

        blank_ratio = round(blank_count / len(sampled_frames), 3)

        # 2. Scene cut detection (histogram comparison)
        scene_cuts: List[float] = []
        prev_hist = None
        for ts, frame in sampled_frames:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0, 1], None, [30, 32], [0, 180, 0, 256])
            cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)

            if prev_hist is not None:
                # Correlation comparison: lower correlation means dramatic change / scene cut
                corr = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL)
                if corr < 0.45:
                    scene_cuts.append(round(ts, 2))
            prev_hist = hist

        scene_cut_rate = round(len(scene_cuts) / duration, 3)

        # 3. Face and Subject Presence
        subject_detected_count = 0
        for ts, frame in sampled_frames:
            h, w = frame.shape[:2]
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces_found = False

            if self.face_cascade:
                faces = self.face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=4,
                    minSize=(int(w * 0.06), int(h * 0.06))
                )
                if len(faces) > 0:
                    faces_found = True

            if not faces_found:
                # Fallback: check central person silhouette presence via edge contrast
                center_crop = gray[int(h * 0.2):int(h * 0.8), int(w * 0.25):int(w * 0.75)]
                if np.std(center_crop) > 30.0:
                    faces_found = True

            if faces_found:
                subject_detected_count += 1

        subject_presence_ratio = round(subject_detected_count / len(sampled_frames), 3)

        # 4. Burned-in subtitle detection in bottom 30% of frames
        burned_sub_votes = 0
        min_x1, min_y1, max_x2, max_y2 = 1.0, 1.0, 0.0, 0.0

        for ts, frame in sampled_frames:
            h, w = frame.shape[:2]
            y_start = int(h * 0.70)
            roi = frame[y_start:, :]
            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray_roi, 80, 200)
            density = float(np.sum(edges > 0)) / float(edges.size)

            # Subtitle text edge density characteristic
            if 0.02 <= density <= 0.22:
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
                closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
                col_proj = np.sum(closed > 0, axis=0)
                active_cols = np.where(col_proj > (roi.shape[0] * 0.1))[0]
                if len(active_cols) > int(w * 0.15):
                    burned_sub_votes += 1
                    min_x1 = min(min_x1, active_cols[0] / float(w))
                    max_x2 = max(max_x2, active_cols[-1] / float(w))
                    min_y1 = min(min_y1, 0.70)
                    max_y2 = max(max_y2, 0.96)

        has_burned_subtitles = (burned_sub_votes / len(sampled_frames)) >= 0.35
        burned_region = None
        if has_burned_subtitles:
            burned_region = {
                "x1": round(max(0.0, min_x1), 3),
                "y1": round(min_y1, 3),
                "x2": round(min(1.0, max_x2), 3),
                "y2": round(max_y2, 3),
            }

        # 5. Evaluate Rejection Criteria
        rejection_reasons: List[str] = []
        if blank_ratio > self.MAX_BLANK_FRAME_RATIO:
            rejection_reasons.append(
                f"Too many blank/black frames ({blank_ratio * 100:.1f}% > {self.MAX_BLANK_FRAME_RATIO * 100:.0f}%)"
            )

        if subject_presence_ratio < self.MIN_SUBJECT_PRESENCE_RATIO:
            rejection_reasons.append(
                f"Subject missing for >30% of duration (presence {subject_presence_ratio * 100:.1f}% < {self.MIN_SUBJECT_PRESENCE_RATIO * 100:.0f}%)"
            )

        if scene_cut_rate > self.MAX_SCENE_CUT_RATE:
            rejection_reasons.append(
                f"Scene cuts too chaotic ({scene_cut_rate:.2f} cuts/sec > {self.MAX_SCENE_CUT_RATE} cuts/sec)"
            )

        is_viable = len(rejection_reasons) == 0

        return VisualAnalysisReport(
            video_path=video_path,
            start_sec=start_sec,
            end_sec=end_sec,
            duration=duration,
            scene_cuts=scene_cuts,
            scene_cut_rate_per_sec=scene_cut_rate,
            blank_frame_ratio=blank_ratio,
            subject_presence_ratio=subject_presence_ratio,
            has_burned_subtitles=has_burned_subtitles,
            burned_subtitle_region=burned_region,
            is_viable=is_viable,
            rejection_reasons=rejection_reasons
        )
