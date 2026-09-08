"""Visual Framing Module: Face Detection Adapter and Smooth Crop Keyframe Generation.

Transforms 16:9 source frames into smooth 9:16 portrait crop keyframes centered on speakers.
Uses OpenCV YuNet ONNX face detection (lightweight CPU inference).
Falls back gracefully to center crop or blurred fallback if no face is detected with sufficient confidence.
"""

import os
import logging
import subprocess
from pathlib import Path
from typing import List, Tuple, Optional
import cv2
import numpy as np

from enum import Enum
from edit_plan import CropKeyframe, FramingMode
from config import settings

logger = logging.getLogger(__name__)

# Default YuNet ONNX model path
DEFAULT_YUNET_PATH = settings.PROJECT_ROOT / "assets" / "models" / "face_detection_yunet_2023mar.onnx"


class SubjectPresenceState(str, Enum):
    VALID_SUBJECT = "VALID_SUBJECT"
    TEMPORARY_FACE_LOSS = "TEMPORARY_FACE_LOSS"
    NO_VALID_SUBJECT = "NO_VALID_SUBJECT"


class VisualFramingAnalyzer:
    """Extracts frames from source video and calculates smoothed portrait crop coordinates."""

    def __init__(self, model_path: Optional[Path] = None):
        self.model_path = model_path or DEFAULT_YUNET_PATH
        self._detector = None

    def _get_detector(self, input_size: Tuple[int, int] = (320, 320)):
        if self._detector is None:
            if not self.model_path.exists():
                raise FileNotFoundError(f"YuNet ONNX model not found at {self.model_path}")
            # FaceDetectorYN expects (width, height)
            create_fn = getattr(cv2, "FaceDetectorYN_create", None)
            if create_fn is None:
                raise RuntimeError("OpenCV FaceDetectorYN is not available in current environment")
            self._detector = create_fn(
                str(self.model_path),
                "",
                input_size,
                score_threshold=0.6,
                nms_threshold=0.3,
                top_k=5000
            )
        else:
            self._detector.setInputSize(input_size)
        return self._detector

    def detect_scene_cuts(
        self,
        video_path: str,
        start_sec: float,
        end_sec: float,
        threshold: float = 28.0
    ) -> List[float]:
        """
        Detects hard camera scene cuts within [start_sec, end_sec].
        Uses fast frame downscaling and mean absolute pixel difference.
        Returns sorted list of clip-local timestamps (seconds from clip start) where cuts occur.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return []

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        start_frame = int(start_sec * fps)
        end_frame = int(end_sec * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        cuts: List[float] = []
        prev_gray = None
        frame_idx = start_frame

        while frame_idx <= end_frame:
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            small = cv2.resize(gray, (160, 90))

            if prev_gray is not None:
                diff = cv2.absdiff(small, prev_gray)
                mean_diff = float(np.mean(diff))
                if mean_diff > threshold:
                    clip_local_t = round((frame_idx - start_frame) / fps, 2)
                    # Deduplicate cuts within 0.5s
                    if not cuts or (clip_local_t - cuts[-1] >= 0.5):
                        cuts.append(clip_local_t)

            prev_gray = small
            frame_idx += 1

        cap.release()
        return cuts

    def analyze_clip_framing(
        self,
        video_path: str,
        start_sec: float,
        end_sec: float,
        sample_interval_sec: float = 1.0,
        aspect_ratio: float = 9.0 / 16.0
    ) -> Tuple[FramingMode, List[CropKeyframe]]:
        """
        Samples frames every `sample_interval_sec` across [start_sec, end_sec].
        Detects faces, calculates bounding box for head-and-upper-body framing,
        applies moving average smoothing, and clamps to source bounds.

        Returns (FramingMode, List[CropKeyframe]).
        If face detection confidence is too low or fails, returns (FramingMode.BLURRED_FALLBACK, []).
        """
        duration = end_sec - start_sec
        if duration <= 0:
            return FramingMode.BLURRED_FALLBACK, []

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.warning(f"Could not open video {video_path} for visual framing analysis.")
            return FramingMode.BLURRED_FALLBACK, []

        src_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        src_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

        if src_width <= 0 or src_height <= 0:
            cap.release()
            return FramingMode.BLURRED_FALLBACK, []

        # Detect scene cuts to enforce segmented tracking timeline
        scene_cuts = self.detect_scene_cuts(video_path, start_sec, end_sec)
        logger.info(f"Visual framing detected {len(scene_cuts)} scene cuts: {scene_cuts}")

        raw_keyframes: List[CropKeyframe] = []
        current_t = start_sec

        try:
            detector = self._get_detector((src_width, src_height))
        except Exception as e:
            logger.warning(f"Face detector initialization failed: {e}. Falling back to blurred composition.")
            cap.release()
            return FramingMode.BLURRED_FALLBACK, []

        # Subject Presence Gate tracking variables:
        # Hysteresis: Hold last valid framing for up to grace_period_sec (2.0s) inside the same scene segment.
        grace_period_sec = 2.0
        last_valid_kf: Optional[CropKeyframe] = None
        consecutive_loss_time = 0.0
        current_scene_idx = 0
        sorted_cuts = sorted(scene_cuts)

        while current_t < end_sec:
            clip_local_t = round(current_t - start_sec, 2)

            # Check if current sample crossed a hard scene cut
            # On scene cut, reset hysteresis instantly (no holdover across scene cuts)
            if current_scene_idx < len(sorted_cuts) and clip_local_t >= sorted_cuts[current_scene_idx]:
                last_valid_kf = None
                consecutive_loss_time = 0.0
                while current_scene_idx < len(sorted_cuts) and clip_local_t >= sorted_cuts[current_scene_idx]:
                    current_scene_idx += 1

            frame_num = int(current_t * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            if not ret or frame is None:
                current_t += sample_interval_sec
                continue

            # Detect face
            try:
                detector.setInputSize((frame.shape[1], frame.shape[0]))
                _, faces = detector.detect(frame)
            except Exception as e:
                logger.debug(f"Frame detection error at {current_t}s: {e}")
                faces = None

            detected_kf: Optional[CropKeyframe] = None
            if faces is not None and len(faces) > 0:
                # Find dominant face (highest confidence or largest area)
                # YuNet format: [x1, y1, w, h, x_re, y_re, x_le, y_le, x_nt, y_nt, x_rc, y_rc, x_lc, y_lc, score]
                best_face = None
                best_score = -1.0
                for f in faces:
                    score = float(f[14])
                    area = float(f[2] * f[3])
                    # Weight area and score
                    combined_metric = score * np.sqrt(area)
                    if combined_metric > best_score:
                        best_score = combined_metric
                        best_face = f

                if best_face is not None and best_face[14] >= 0.5:
                    fx, fy, fw, fh = best_face[0], best_face[1], best_face[2], best_face[3]
                    # Head-and-upper-body center: face center X, face center Y shifted slightly down (to include neck/shoulders)
                    face_center_x = fx + fw / 2.0
                    face_center_y = fy + fh / 2.0 + fh * 0.3  # Offset down slightly for natural upper body framing

                    norm_x = float(np.clip(face_center_x / src_width, 0.0, 1.0))
                    norm_y = float(np.clip(face_center_y / src_height, 0.0, 1.0))

                    detected_kf = CropKeyframe(
                        time=clip_local_t,
                        crop_center_x=round(norm_x, 4),
                        crop_center_y=round(norm_y, 4),
                        confidence=round(float(best_face[14]), 3)
                    )

            # Subject Presence Gate evaluation:
            if detected_kf is not None:
                # VALID_SUBJECT
                last_valid_kf = detected_kf
                consecutive_loss_time = 0.0
                raw_keyframes.append(detected_kf)
            else:
                consecutive_loss_time += sample_interval_sec
                if last_valid_kf is not None and consecutive_loss_time <= grace_period_sec:
                    # TEMPORARY_FACE_LOSS -> HOLD previous framing (do NOT jump or fall to empty frame)
                    held_kf = CropKeyframe(
                        time=clip_local_t,
                        crop_center_x=last_valid_kf.crop_center_x,
                        crop_center_y=last_valid_kf.crop_center_y,
                        confidence=round(max(0.4, last_valid_kf.confidence - 0.1), 3)
                    )
                    raw_keyframes.append(held_kf)
                    logger.debug(f"t={clip_local_t}s: TEMPORARY_FACE_LOSS -> holding previous framing ({held_kf.crop_center_x})")
                else:
                    # NO_VALID_SUBJECT
                    logger.debug(f"t={clip_local_t}s: NO_VALID_SUBJECT")

            current_t += sample_interval_sec

        cap.release()

        # Check coverage: if faces detected in fewer than 35% of sample points, fall back
        expected_samples = max(1, int(duration / sample_interval_sec))
        detection_ratio = len(raw_keyframes) / float(expected_samples)
        logger.info(
            f"Visual framing detected faces in {len(raw_keyframes)}/{expected_samples} "
            f"samples ({detection_ratio * 100:.1f}%)"
        )

        if detection_ratio < 0.35 or not raw_keyframes:
            logger.info("Face detection coverage below confidence threshold -> BLURRED_FALLBACK")
            return FramingMode.BLURRED_FALLBACK, []

        # Smooth crop keyframes with segmented scene-cut boundaries
        # Forbids interpolation/smoothing across camera scene cuts
        smoothed = self.smooth_segmented_keyframes(
            raw_keyframes,
            scene_cuts=scene_cuts,
            src_width=src_width,
            src_height=src_height,
            aspect_ratio=aspect_ratio
        )
        return FramingMode.FACE_TRACKED, smoothed

    @classmethod
    def smooth_segmented_keyframes(
        cls,
        keyframes: List[CropKeyframe],
        scene_cuts: List[float],
        src_width: int,
        src_height: int,
        aspect_ratio: float = 9.0 / 16.0,
        window_size: int = 3
    ) -> List[CropKeyframe]:
        """
        Segments timeline by scene cuts.
        Applies temporal smoothing independently within each scene segment.
        Strictly forbids any cross-boundary smoothing/interpolation between scenes.
        """
        if not keyframes:
            return []

        if not scene_cuts:
            return cls.smooth_keyframes(keyframes, src_width, src_height, aspect_ratio, window_size)

        # Build segments: [0.0, cut1], (cut1, cut2], ... (cutN, inf)
        cuts = sorted(scene_cuts)
        segments: List[List[CropKeyframe]] = []
        current_segment: List[CropKeyframe] = []

        cut_idx = 0
        for k in keyframes:
            # Check if this keyframe crosses into next scene
            while cut_idx < len(cuts) and k.time >= cuts[cut_idx]:
                if current_segment:
                    segments.append(current_segment)
                    current_segment = []
                cut_idx += 1
            current_segment.append(k)

        if current_segment:
            segments.append(current_segment)

        # Smooth each scene independently
        all_smoothed: List[CropKeyframe] = []
        for seg in segments:
            smoothed_seg = cls.smooth_keyframes(seg, src_width, src_height, aspect_ratio, window_size)
            all_smoothed.extend(smoothed_seg)

        return all_smoothed

    @staticmethod
    def smooth_keyframes(
        keyframes: List[CropKeyframe],
        src_width: int,
        src_height: int,
        aspect_ratio: float = 9.0 / 16.0,
        window_size: int = 3,
        max_displacement_x_per_sec: float = 0.08,
        max_displacement_y_per_sec: float = 0.04,
    ) -> List[CropKeyframe]:
        """
        Applies temporal moving average to crop center coordinates to eliminate jitter.
        Enforces intra-shot motion limits (clamping max displacement per second) so the camera
        glides smoothly and never jerks/jumps abruptly within the same shot.
        Ensures the 9:16 portrait crop window remains strictly within source bounds.
        """
        if not keyframes:
            return []

        # Target crop window in source pixels: height = src_height, width = src_height * (9/16)
        crop_w_px = src_height * aspect_ratio
        half_crop_norm_x = (crop_w_px / 2.0) / src_width

        # Safe bounds for crop center X so the box never exceeds [0, src_width]
        min_cx = half_crop_norm_x
        max_cx = 1.0 - half_crop_norm_x

        xs = [k.crop_center_x for k in keyframes]
        ys = [k.crop_center_y for k in keyframes]

        smoothed_keyframes: List[CropKeyframe] = []
        n = len(keyframes)

        prev_x = None
        prev_y = None
        prev_time = None

        for i in range(n):
            start_idx = max(0, i - window_size // 2)
            end_idx = min(n, i + window_size // 2 + 1)
            mean_x = float(np.mean(xs[start_idx:end_idx]))
            mean_y = float(np.mean(ys[start_idx:end_idx]))

            # Clamp mean_x within safe crop boundary if crop fits in source
            if min_cx < max_cx:
                clamped_x = float(np.clip(mean_x, min_cx, max_cx))
            else:
                clamped_x = 0.5

            clamped_y = float(np.clip(mean_y, 0.2, 0.8))

            cur_time = keyframes[i].time
            # Apply motion limit clamping relative to previous keyframe in the same segment
            if prev_x is not None and prev_y is not None and prev_time is not None:
                dt = max(0.1, cur_time - prev_time)
                max_dx = max_displacement_x_per_sec * dt
                max_dy = max_displacement_y_per_sec * dt

                clamped_x = float(np.clip(clamped_x, prev_x - max_dx, prev_x + max_dx))
                clamped_y = float(np.clip(clamped_y, prev_y - max_dy, prev_y + max_dy))

            prev_x = clamped_x
            prev_y = clamped_y
            prev_time = cur_time

            smoothed_keyframes.append(CropKeyframe(
                time=cur_time,
                crop_center_x=round(clamped_x, 4),
                crop_center_y=round(clamped_y, 4),
                confidence=keyframes[i].confidence
            ))

        return smoothed_keyframes


visual_framing = VisualFramingAnalyzer()
