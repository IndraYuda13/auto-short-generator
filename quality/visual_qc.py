"""Visual Quality Control (QC) module for rendered short-form videos.

Implements Blueprint Bab 16.2:
- Samples frames across clip (e.g. every 2s, minimum 10 frames)
- Checks for blank/black/white frames (mean pixel intensity < 5 or > 250)
- Checks for black/white flashes
- Checks for missing subject (ratio of frames without detected subject/contour)
- Checks for badly cut faces (face bounding box truncated at top/bottom without headroom)
- Checks for subtitle overlap and duplicate/stuck subtitles across consecutive frames
- Checks boundary safe-zone (subtitles must NOT be in top 15% or bottom 20% danger zones)
- Exports typed Pydantic VisualQCResult data contract
"""

import logging
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
import cv2
import numpy as np
from pydantic import BaseModel, Field

from config import settings

logger = logging.getLogger(__name__)

DEFAULT_YUNET_PATH = settings.PROJECT_ROOT / "assets" / "models" / "face_detection_yunet_2023mar.onnx"
DEFAULT_CASCADE_PATH = settings.PROJECT_ROOT / "assets" / "models" / "haarcascade_frontalface_default.xml"


class VisualQCResult(BaseModel):
    """Structured result of Visual Quality Control evaluation (Blueprint Bab 16.2)."""
    passed: bool = Field(..., description="True if all visual criteria pass without error")
    sampled_frames_count: int = Field(default=0, description="Total number of frames sampled across clip")
    blank_frames: int = Field(default=0, description="Count of blank/black/white frames detected")
    subject_present_ratio: float = Field(default=0.0, description="Ratio of frames containing detected primary subject")
    subtitle_safe: bool = Field(default=True, description="True if subtitles comply with boundary safe-zones")
    errors: List[str] = Field(default_factory=list, description="List of visual defect descriptions")


class VisualQC:
    """Automated visual quality inspector for vertical short-form video."""

    DEFAULT_SAMPLE_INTERVAL_SEC: float = 2.0
    DEFAULT_MIN_FRAMES: int = 10
    DEFAULT_MIN_SUBJECT_RATIO: float = 0.70
    DANGER_TOP_RATIO: float = 0.15      # Top 15% is danger zone for subtitles
    DANGER_BOTTOM_RATIO: float = 0.20   # Bottom 20% is danger zone for subtitles

    def __init__(
        self,
        sample_interval_sec: float = DEFAULT_SAMPLE_INTERVAL_SEC,
        min_frames: int = DEFAULT_MIN_FRAMES,
        min_subject_ratio: float = DEFAULT_MIN_SUBJECT_RATIO,
        model_path: Optional[Path] = None,
        cascade_path: Optional[Path] = None,
    ):
        self.sample_interval_sec = float(sample_interval_sec)
        self.min_frames = int(min_frames)
        self.min_subject_ratio = float(min_subject_ratio)
        self.model_path = Path(model_path or DEFAULT_YUNET_PATH)
        self.cascade_path = Path(cascade_path or DEFAULT_CASCADE_PATH)

        self._yunet_detector = None
        self._cascade_detector = None
        self._init_detectors()

    def _init_detectors(self):
        """Initializes YuNet face detector with fallback to Haar cascade."""
        yn_create = getattr(cv2, "FaceDetectorYN_create", None)
        if self.model_path.exists() and yn_create is not None:
            try:
                self._yunet_detector = yn_create(
                    str(self.model_path),
                    "",
                    (320, 320),
                    score_threshold=0.50,
                    nms_threshold=0.3,
                    top_k=10,
                )
            except Exception as e:
                logger.warning(f"Failed to initialize YuNet face detector: {e}")

        cascade_cls = getattr(cv2, "CascadeClassifier", None)
        if not self._yunet_detector and cascade_cls is not None and self.cascade_path.exists():
            try:
                self._cascade_detector = cascade_cls(str(self.cascade_path))
            except Exception as e:
                logger.warning(f"Failed to initialize Haar cascade: {e}")

    def evaluate(self, video_path: Union[str, Path]) -> VisualQCResult:
        """Performs full visual QC on the given video file."""
        path = Path(video_path)
        if not path.exists():
            return VisualQCResult(
                passed=False,
                sampled_frames_count=0,
                blank_frames=0,
                subject_present_ratio=0.0,
                subtitle_safe=False,
                errors=[f"Video file does not exist: {path}"]
            )

        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            return VisualQCResult(
                passed=False,
                sampled_frames_count=0,
                blank_frames=0,
                subject_present_ratio=0.0,
                subtitle_safe=False,
                errors=["Cannot open video stream with OpenCV"]
            )

        try:
            sampled_items = self._sample_frames(cap)
        finally:
            cap.release()

        if not sampled_items:
            return VisualQCResult(
                passed=False,
                sampled_frames_count=0,
                blank_frames=0,
                subject_present_ratio=0.0,
                subtitle_safe=False,
                errors=["Could not extract any valid frames from video"]
            )

        return self.evaluate_frames(sampled_items)

    def _sample_frames(self, cap: cv2.VideoCapture) -> List[Tuple[float, np.ndarray]]:
        """Samples frames across the clip adhering to sample_interval_sec and min_frames."""
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if total_frames > 0 else 0.0

        if duration <= 0.0:
            return []

        # Determine number of frames to sample
        interval_count = int(duration / self.sample_interval_sec)
        target_count = max(self.min_frames, interval_count)
        # Bounded by total available frames
        target_count = min(target_count, max(1, total_frames))

        step_sec = duration / (target_count + 1)
        sampled: List[Tuple[float, np.ndarray]] = []

        for i in range(1, target_count + 1):
            t = round(i * step_sec, 3)
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
            ret, frame = cap.read()
            if ret and frame is not None and frame.size > 0:
                sampled.append((t, frame))

        return sampled

    def evaluate_frames(self, sampled_items: List[Tuple[float, np.ndarray]]) -> VisualQCResult:
        """Evaluates a pre-sampled list of (timestamp_sec, frame_bgr) tuples."""
        sampled_count = len(sampled_items)
        if sampled_count == 0:
            return VisualQCResult(
                passed=False,
                sampled_frames_count=0,
                blank_frames=0,
                subject_present_ratio=0.0,
                subtitle_safe=False,
                errors=["Empty frame set provided for visual QC"]
            )

        errors: List[str] = []

        # 1 & 2: Blank frames and flashes
        blank_frames, blank_errors = self.check_blank_and_flashes(sampled_items)
        errors.extend(blank_errors)

        # 3 & 4: Subject presence and face framing
        subject_ratio, face_framing_errors = self.check_subject_and_face_framing(sampled_items)
        errors.extend(face_framing_errors)

        # 5: Subtitle overlap & duplicate / stuck subtitles
        subtitle_overlap_errors = self.check_subtitle_overlap_and_duplicates(sampled_items)
        errors.extend(subtitle_overlap_errors)

        # 6: Boundary safe-zone check
        subtitle_safe, safe_zone_errors = self.check_boundary_safe_zone(sampled_items)
        errors.extend(safe_zone_errors)

        passed = (
            len(errors) == 0
            and blank_frames == 0
            and subject_ratio >= self.min_subject_ratio
            and subtitle_safe
        )

        return VisualQCResult(
            passed=passed,
            sampled_frames_count=sampled_count,
            blank_frames=blank_frames,
            subject_present_ratio=round(subject_ratio, 3),
            subtitle_safe=subtitle_safe,
            errors=errors,
        )

    def check_blank_and_flashes(
        self,
        sampled_items: List[Tuple[float, np.ndarray]],
    ) -> Tuple[int, List[str]]:
        """Checks for blank/black/white frames (mean < 5 or > 250) and sudden flashes."""
        blank_frames = 0
        mean_intensities: List[float] = []
        errors: List[str] = []

        for t, frame in sampled_items:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            mean_val = float(np.mean(gray))
            mean_intensities.append(mean_val)

            # Check blank frame (< 5 is black, > 250 is white blown out)
            if mean_val < 5.0 or mean_val > 250.0:
                blank_frames += 1

        if blank_frames > 0:
            errors.append(
                f"Detected {blank_frames} blank/black/white frames (mean pixel intensity < 5 or > 250)"
            )

        # Check black/white flashes (abrupt isolated spike or drop)
        n = len(mean_intensities)
        flash_timestamps: List[float] = []
        for i in range(1, n - 1):
            curr = mean_intensities[i]
            prev_m = mean_intensities[i - 1]
            next_m = mean_intensities[i + 1]

            # Isolated black flash between normal frames
            is_black_flash = (curr < 5.0 and prev_m > 20.0 and next_m > 20.0)
            # Isolated white flash between normal frames
            is_white_flash = (curr > 250.0 and prev_m < 230.0 and next_m < 230.0)
            # Sudden high-contrast luminance spike and immediate return
            is_spike_flash = (
                abs(curr - prev_m) > 120.0
                and abs(curr - next_m) > 120.0
                and abs(prev_m - next_m) < 60.0
            )

            if is_black_flash or is_white_flash or is_spike_flash:
                t_flash = sampled_items[i][0]
                flash_timestamps.append(t_flash)

        if flash_timestamps:
            times_str = ", ".join(f"{ts:.1f}s" for ts in flash_timestamps[:3])
            errors.append(f"Detected black/white flash at {times_str}")

        return blank_frames, errors

    def check_subject_and_face_framing(
        self,
        sampled_items: List[Tuple[float, np.ndarray]],
    ) -> Tuple[float, List[str]]:
        """Checks subject presence ratio and verifies face bounding boxes are not cut badly."""
        subject_count = 0
        errors: List[str] = []
        badly_cut_count = 0

        for t, frame in sampled_items:
            h, w = frame.shape[:2]
            faces = self.detect_faces(frame)

            subject_found = False

            if faces:
                subject_found = True
                # Check each detected face for boundary truncation (face cut badly)
                for (fx, fy, fw, fh, conf) in faces:
                    # Top cut check: face touches or exceeds top canvas boundary without headroom
                    top_truncated = (fy <= 0 or fy < int(0.02 * h))
                    # Bottom cut check: face extends into canvas bottom boundary
                    bottom_truncated = (fy + fh >= (h - 5) or (fy + fh) > int(0.98 * h))

                    if top_truncated or bottom_truncated:
                        badly_cut_count += 1
                        cut_type = "top (no headroom)" if top_truncated else "bottom edge"
                        if badly_cut_count <= 2:
                            errors.append(
                                f"Face cut badly at {cut_type} at t={t:.1f}s "
                                f"(box y={fy}, h={fh} on {w}x{h} canvas)"
                            )

            if not subject_found:
                # Foreground contour / central energy fallback
                if self._check_central_foreground(frame):
                    subject_found = True

            if subject_found:
                subject_count += 1

        total = len(sampled_items)
        ratio = subject_count / total if total > 0 else 0.0

        if ratio < self.min_subject_ratio:
            errors.append(
                f"Subject missing in excessive frames: presence ratio {ratio:.2f} "
                f"is below required threshold {self.min_subject_ratio:.2f}"
            )

        return ratio, errors

    def _check_central_foreground(self, frame: np.ndarray) -> bool:
        """Determines if there is significant foreground subject/contour in the central vertical zone."""
        h, w = frame.shape[:2]
        # Central crop: x from 20% to 80%, y from 10% to 85%
        x1, x2 = int(0.20 * w), int(0.80 * w)
        y1, y2 = int(0.10 * h), int(0.85 * h)
        central = frame[y1:y2, x1:x2]

        if central.size == 0:
            return False

        gray = cv2.cvtColor(central, cv2.COLOR_BGR2GRAY)
        # Edge detection
        edges = cv2.Canny(gray, 40, 120)
        edge_density = float(np.count_nonzero(edges)) / float(edges.size)
        std_intensity = float(np.std(gray))

        # Check for significant contour
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        central_area = central.shape[0] * central.shape[1]
        has_large_contour = any(cv2.contourArea(c) > (0.04 * central_area) for c in contours)

        return (edge_density > 0.015 and std_intensity > 22.0) or has_large_contour

    def detect_faces(self, frame: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
        """Detects faces in frame using YuNet (or fallback Haar cascade)."""
        h, w = frame.shape[:2]
        detected: List[Tuple[int, int, int, int, float]] = []

        if self._yunet_detector is not None:
            try:
                self._yunet_detector.setInputSize((w, h))
                ret, faces = self._yunet_detector.detect(frame)
                if ret and faces is not None:
                    for f in faces:
                        fx = max(0, int(f[0]))
                        fy = max(0, int(f[1]))
                        fw = min(w - fx, int(f[2]))
                        fh = min(h - fy, int(f[3]))
                        conf = float(f[14])
                        if fw > 20 and fh > 20 and conf >= 0.45:
                            detected.append((fx, fy, fw, fh, conf))
                    return detected
            except Exception as e:
                logger.debug(f"YuNet inference error: {e}")

        if self._cascade_detector is not None:
            try:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self._cascade_detector.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30)
                )
                for (fx, fy, fw, fh) in faces:
                    detected.append((int(fx), int(fy), int(fw), int(fh), 0.80))
            except Exception as e:
                logger.debug(f"Haar cascade detection error: {e}")

        return detected

    def check_subtitle_overlap_and_duplicates(
        self,
        sampled_items: List[Tuple[float, np.ndarray]],
    ) -> List[str]:
        """Checks for overlapping subtitles and duplicate/stuck subtitles across consecutive frames."""
        errors: List[str] = []
        subtitle_rois: List[Tuple[float, Optional[np.ndarray]]] = []

        # Extract subtitle zone: y from 55% to 85% of height
        for t, frame in sampled_items:
            h, w = frame.shape[:2]
            y1, y2 = int(0.55 * h), int(0.85 * h)
            sub_roi = frame[y1:y2, :]

            # Check if this ROI contains text
            text_present, edge_map = self._extract_text_edges(sub_roi)
            if text_present:
                subtitle_rois.append((t, edge_map))
            else:
                subtitle_rois.append((t, None))

        # Check for stuck / duplicate subtitles (> 4 consecutive sampled frames with identical text)
        consecutive_duplicates = 0
        max_duplicates = 0
        last_edge_map = None

        for t, edge_map in subtitle_rois:
            if edge_map is not None:
                if last_edge_map is not None and edge_map.shape == last_edge_map.shape:
                    # Normalized correlation or diff
                    diff = cv2.absdiff(edge_map, last_edge_map)
                    diff_ratio = float(np.count_nonzero(diff)) / float(diff.size)
                    if diff_ratio < 0.015:  # Almost perfectly identical
                        consecutive_duplicates += 1
                        if consecutive_duplicates > max_duplicates:
                            max_duplicates = consecutive_duplicates
                    else:
                        consecutive_duplicates = 0
                else:
                    consecutive_duplicates = 0
                last_edge_map = edge_map
            else:
                consecutive_duplicates = 0
                last_edge_map = None

        # 4 consecutive samples at 2s interval = ~8 seconds of identical stuck subtitle
        if max_duplicates >= 4:
            errors.append(
                f"Duplicate or stuck subtitle detected across {max_duplicates + 1} consecutive frames "
                "(>8s without update)"
            )

        # Check for overlapping subtitles in any frame
        for t, frame in sampled_items:
            h, w = frame.shape[:2]
            y1, y2 = int(0.55 * h), int(0.85 * h)
            sub_roi = frame[y1:y2, :]

            if self._detect_overlapping_text_boxes(sub_roi):
                errors.append(
                    f"Subtitle overlap detected: conflicting text lines or colliding subtitle layers at t={t:.1f}s"
                )
                break

        return errors

    def _extract_text_edges(self, roi: np.ndarray) -> Tuple[bool, Optional[np.ndarray]]:
        """Extracts high-contrast edge features indicative of subtitle text in a region."""
        if roi is None or roi.size == 0:
            return False, None

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        # Subtitle text typically has high contrast edges
        edges = cv2.Canny(gray, 80, 200)

        # Text consists of multiple character glyphs/strokes
        contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        glyph_contours = [
            c for c in contours
            if 8 <= cv2.boundingRect(c)[3] <= 110 and 4 <= cv2.boundingRect(c)[2] <= 150
        ]

        # If at least 4 text-like character glyphs are present
        if len(glyph_contours) >= 4:
            return True, edges
        return False, None

    def _detect_overlapping_text_boxes(self, roi: np.ndarray) -> bool:
        """Detects if multiple text lines or bounding boxes overlap vertically."""
        if roi is None or roi.size == 0:
            return False

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        # Morphological operation to group text characters into horizontal lines
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 3))
        grad = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)
        _, thresh = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        # Dilate horizontally to connect words into line boxes
        line_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (35, 5))
        connected = cv2.dilate(thresh, line_kernel, iterations=1)

        contours, _ = cv2.findContours(connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        text_boxes: List[Tuple[int, int, int, int]] = []

        roi_w = roi.shape[1]
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            # Text line box criteria: wide aspect ratio and reasonable height
            if w > (0.15 * roi_w) and 15 < h < 140:
                text_boxes.append((x, y, w, h))

        # Check for vertical overlaps between bounding boxes
        n = len(text_boxes)
        for i in range(n):
            for j in range(i + 1, n):
                x1, y1, w1, h1 = text_boxes[i]
                x2, y2, w2, h2 = text_boxes[j]

                # Check horizontal overlap
                h_overlap = not (x1 + w1 < x2 or x2 + w2 < x1)
                if h_overlap:
                    # Check vertical overlap (collision)
                    # If boxes intersect vertically by more than 10 pixels
                    overlap_y = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
                    if overlap_y > 10:
                        return True

        return False

    def check_boundary_safe_zone(
        self,
        sampled_items: List[Tuple[float, np.ndarray]],
    ) -> Tuple[bool, List[str]]:
        """Verifies subtitles do not violate safe zones (top 15% or bottom 20% danger areas)."""
        errors: List[str] = []
        top_violations = 0
        bottom_violations = 0

        for t, frame in sampled_items:
            h, w = frame.shape[:2]
            top_danger_h = int(self.DANGER_TOP_RATIO * h)      # Top 15%
            bottom_danger_y = int((1.0 - self.DANGER_BOTTOM_RATIO) * h)  # Bottom 20%

            top_strip = frame[:top_danger_h, :]
            bottom_strip = frame[bottom_danger_y:, :]

            if self._has_subtitle_text_in_strip(top_strip):
                top_violations += 1
            if self._has_subtitle_text_in_strip(bottom_strip):
                bottom_violations += 1

        # Violation threshold: if text appears in danger area in more than 1 sampled frame
        if top_violations > 1:
            errors.append(
                f"Subtitle violates boundary safe-zone: text detected in top 15% danger area "
                f"({top_violations} frames)"
            )
        if bottom_violations > 1:
            errors.append(
                f"Subtitle violates boundary safe-zone: text detected in bottom 20% danger area "
                f"({bottom_violations} frames)"
            )

        subtitle_safe = (top_violations <= 1 and bottom_violations <= 1)
        return subtitle_safe, errors

    def _has_subtitle_text_in_strip(self, strip: np.ndarray) -> bool:
        """Detects if an image strip contains prominent horizontal subtitle text."""
        if strip is None or strip.size == 0:
            return False

        h, w = strip.shape[:2]
        gray = cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY)

        # High-contrast edge detection for text
        edges = cv2.Canny(gray, 100, 220)

        # Group horizontal character strokes
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 3))
        grouped = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(grouped, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            x, y, cw, ch = cv2.boundingRect(c)
            # Text bar criteria: wide aspect ratio (aspect > 2.5), width > 18% of frame, height 18-90px
            if cw > (0.18 * w) and 18 <= ch <= 95 and (cw / max(1, ch)) > 2.5:
                # Confirm edge density inside box
                box_edges = edges[y:y+ch, x:x+cw]
                if box_edges.size > 0:
                    density = float(np.count_nonzero(box_edges)) / float(box_edges.size)
                    if density > 0.08:
                        return True

        return False


def evaluate_visual_qc(
    video_path: Union[str, Path],
    sample_interval_sec: float = VisualQC.DEFAULT_SAMPLE_INTERVAL_SEC,
    min_frames: int = VisualQC.DEFAULT_MIN_FRAMES,
    min_subject_ratio: float = VisualQC.DEFAULT_MIN_SUBJECT_RATIO,
) -> VisualQCResult:
    """Convenience function to evaluate visual QC on a video file."""
    qc = VisualQC(
        sample_interval_sec=sample_interval_sec,
        min_frames=min_frames,
        min_subject_ratio=min_subject_ratio,
    )
    return qc.evaluate(video_path)
