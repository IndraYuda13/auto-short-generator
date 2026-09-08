"""Scene-Static Portrait Framing Module (Blueprint Bab 12).

Enforces:
1. Scene-static portrait framing:
   - Scene 1: detect face/person -> calculate best 9:16 crop -> HOLD
   - Scene cut -> Scene 2: detect again -> new crop -> HOLD
   - ZERO continuous camera tracking! (No camera jitter, pan, or float)
2. Single speaker:
   - Head + upper torso framing
   - Headroom consistency
   - Natural framing (wajah tidak terlalu dekat / no aggressive digital zoom)
3. Multi speaker:
   - JANGAN active-speaker switching!
   - Safe two-person crop if speakers fit within 9:16 canvas
   - If too wide to fit: safe full-frame fallback, or reject candidate
4. Output: List of SceneCrop windows per scene cut segment.
"""

import os
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import cv2
import numpy as np
from pydantic import BaseModel, Field

from editing.edit_plan import SceneCrop
from config import settings

logger = logging.getLogger(__name__)

# Default model paths
DEFAULT_YUNET_PATH = settings.PROJECT_ROOT / "assets" / "models" / "face_detection_yunet_2023mar.onnx"
DEFAULT_CASCADE_PATH = settings.PROJECT_ROOT / "assets" / "models" / "haarcascade_frontalface_default.xml"


class FramingDecision(BaseModel):
    """Decision output for video framing across all scene cuts."""
    layout: str = Field(
        default="PORTRAIT_9_16",
        description="PORTRAIT_9_16, SAFE_FULL_FRAME, or REJECT"
    )
    crop_windows: List[SceneCrop] = Field(default_factory=list)
    speaker_mode: str = Field(
        default="SINGLE_SPEAKER",
        description="SINGLE_SPEAKER, MULTI_SPEAKER_SAFE_TWO_SHOT, MULTI_SPEAKER_WIDE, NO_SPEAKER"
    )
    scene_cuts: List[float] = Field(default_factory=list)
    is_rejected: bool = False
    rejection_reason: Optional[str] = None


class SceneStaticFraming:
    """Computes scene-static framing for short-form clips adhering to Blueprint Bab 12."""

    def __init__(
        self,
        model_path: Optional[Path] = None,
        cascade_path: Optional[Path] = None
    ):
        self.model_path = Path(model_path or DEFAULT_YUNET_PATH)
        self.cascade_path = Path(cascade_path or DEFAULT_CASCADE_PATH)

        self._yunet_detector = None
        self._cascade_detector = None
        self._init_detectors()

    def _init_detectors(self):
        """Initializes YuNet face detector with fallback to Haar Cascade."""
        # 1. Try OpenCV FaceDetectorYN
        yn_create = getattr(cv2, "FaceDetectorYN_create", None)
        if self.model_path.exists() and yn_create is not None:
            try:
                self._yunet_detector = yn_create(
                    str(self.model_path),
                    "",
                    (320, 320),
                    score_threshold=0.55,
                    nms_threshold=0.3,
                    top_k=10
                )
            except Exception as e:
                logger.warning(f"Failed to initialize YuNet: {e}")

        # 2. Try Haar Cascade
        cascade_cls = getattr(cv2, "CascadeClassifier", None)
        if not self._yunet_detector and cascade_cls is not None:
            if not self.cascade_path.exists():
                cv2_data = getattr(cv2, "data", None)
                if cv2_data and hasattr(cv2_data, "haarcascades"):
                    self.cascade_path = Path(cv2_data.haarcascades) / "haarcascade_frontalface_default.xml"

            if self.cascade_path.exists():
                try:
                    self._cascade_detector = cascade_cls(str(self.cascade_path))
                except Exception as e:
                    logger.warning(f"Failed to initialize Haar cascade: {e}")

    def detect_faces(self, frame: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
        """Detects faces in a BGR frame.

        Returns list of tuples: (x, y, w, h, confidence)
        """
        if frame is None or frame.size == 0:
            return []

        h, w = frame.shape[:2]
        detected: List[Tuple[int, int, int, int, float]] = []

        # 1. Try YuNet
        if self._yunet_detector is not None:
            try:
                self._yunet_detector.setInputSize((w, h))
                ret, faces = self._yunet_detector.detect(frame)
                if ret is not None and faces is not None and len(faces) > 0:
                    for f in faces:
                        conf = float(f[14])
                        if conf >= 0.5:
                            fx, fy, fw, fh = int(f[0]), int(f[1]), int(f[2]), int(f[3])
                            # Clamp within frame bounds
                            fx = max(0, min(w - 1, fx))
                            fy = max(0, min(h - 1, fy))
                            fw = max(1, min(w - fx, fw))
                            fh = max(1, min(h - fy, fh))
                            detected.append((fx, fy, fw, fh, conf))
                    if detected:
                        return detected
            except Exception as e:
                logger.debug(f"YuNet detection exception: {e}")

        # 2. Fallback to Haar Cascade
        if self._cascade_detector is not None:
            try:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self._cascade_detector.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=4,
                    minSize=(int(w * 0.05), int(h * 0.05))
                )
                for (fx, fy, fw, fh) in faces:
                    detected.append((int(fx), int(fy), int(fw), int(fh), 0.80))
                return detected
            except Exception as e:
                logger.debug(f"Haar cascade detection exception: {e}")

        return []

    def detect_scene_cuts(
        self,
        video_path: str,
        start_sec: float,
        end_sec: float,
        threshold: float = 28.0
    ) -> List[float]:
        """Detects scene cut timestamps within [start_sec, end_sec].

        Uses fast frame downscaling and mean absolute pixel difference.
        Returns timestamps relative to clip start (0.0 to duration).
        """
        if not os.path.exists(video_path):
            return []

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
                    # Minimum distance between cuts: 0.5s to avoid double-trigger
                    if not cuts or (clip_local_t - cuts[-1] >= 0.5):
                        cuts.append(clip_local_t)

            prev_gray = small
            frame_idx += 1

        cap.release()
        return cuts

    def calculate_scene_crop(
        self,
        faces: List[Tuple[int, int, int, int, float]],
        frame_w: int,
        frame_h: int,
        scene_start: float,
        scene_end: float,
        allow_safe_full_frame: bool = True
    ) -> Tuple[SceneCrop, str, str]:
        """Calculates static crop window for one scene segment.

        Enforces:
        - Single speaker: head + upper torso, consistent headroom, not too close
        - Multi-speaker: NO active-speaker switching. If fits in 9:16 safe two-shot, use it;
          otherwise safe full-frame or reject.
        - Even integer coordinates for FFmpeg compatibility.

        Returns:
            (SceneCrop, layout, speaker_mode)
        """
        # Ensure dimensions are even
        frame_w = (frame_w // 2) * 2
        frame_h = (frame_h // 2) * 2

        # 9:16 portrait crop dimensions inside source height
        target_crop_h = frame_h
        target_crop_w = int(round(target_crop_h * 9.0 / 16.0))
        target_crop_w = (target_crop_w // 2) * 2

        # Ensure crop width does not exceed source width
        if target_crop_w > frame_w:
            target_crop_w = frame_w
            target_crop_h = int(round(target_crop_w * 16.0 / 9.0))
            target_crop_h = (target_crop_h // 2) * 2

        # --- Case 1: Single Speaker ---
        if len(faces) == 1:
            fx, fy, fw, fh, _ = faces[0]
            face_center_x = fx + fw / 2.0

            # Center 9:16 crop horizontally around the face
            crop_x = int(round(face_center_x - target_crop_w / 2.0))
            crop_x = max(0, min(frame_w - target_crop_w, crop_x))
            crop_x = (crop_x // 2) * 2

            # Head + upper torso vertical placement:
            # Full height naturally includes head and upper torso with natural headroom
            crop_y = 0

            crop = SceneCrop(
                scene_start=scene_start,
                scene_end=scene_end,
                crop_x=crop_x,
                crop_y=crop_y,
                crop_w=target_crop_w,
                crop_h=target_crop_h
            )
            return crop, "PORTRAIT_9_16", "SINGLE_SPEAKER"

        # --- Case 2: Multi-Speaker (N >= 2) ---
        elif len(faces) >= 2:
            # Calculate outer bounding box of all speakers
            min_x = min(f[0] for f in faces)
            max_x = max(f[0] + f[2] for f in faces)
            group_span_w = max_x - min_x

            # Safe margin around group
            group_margin = int(target_crop_w * 0.10)
            required_span = group_span_w + 2 * group_margin

            if required_span <= target_crop_w:
                # Safe two-person crop fits inside 9:16!
                group_center_x = (min_x + max_x) / 2.0
                crop_x = int(round(group_center_x - target_crop_w / 2.0))
                crop_x = max(0, min(frame_w - target_crop_w, crop_x))
                crop_x = (crop_x // 2) * 2

                crop = SceneCrop(
                    scene_start=scene_start,
                    scene_end=scene_end,
                    crop_x=crop_x,
                    crop_y=0,
                    crop_w=target_crop_w,
                    crop_h=target_crop_h
                )
                return crop, "PORTRAIT_9_16", "MULTI_SPEAKER_SAFE_TWO_SHOT"
            else:
                # Speakers are too far apart to fit in 9:16 portrait.
                # Rule: JANGAN active-speaker switching! Safe full-frame or reject.
                if allow_safe_full_frame:
                    crop = SceneCrop(
                        scene_start=scene_start,
                        scene_end=scene_end,
                        crop_x=0,
                        crop_y=0,
                        crop_w=frame_w,
                        crop_h=frame_h
                    )
                    return crop, "SAFE_FULL_FRAME", "MULTI_SPEAKER_WIDE"
                else:
                    crop = SceneCrop(
                        scene_start=scene_start,
                        scene_end=scene_end,
                        crop_x=0,
                        crop_y=0,
                        crop_w=frame_w,
                        crop_h=frame_h
                    )
                    return crop, "REJECT", "MULTI_SPEAKER_REJECT"

        # --- Case 3: No Face Detected ---
        else:
            # Fallback: Calm center 9:16 crop
            crop_x = (frame_w - target_crop_w) // 2
            crop_x = (crop_x // 2) * 2

            crop = SceneCrop(
                scene_start=scene_start,
                scene_end=scene_end,
                crop_x=crop_x,
                crop_y=0,
                crop_w=target_crop_w,
                crop_h=target_crop_h
            )
            return crop, "PORTRAIT_9_16", "NO_SPEAKER"

    def analyze_framing(
        self,
        video_path: str,
        start_sec: float = 0.0,
        end_sec: Optional[float] = None,
        scene_cuts: Optional[List[float]] = None,
        allow_safe_full_frame: bool = True,
        default_layout: str = "SAFE_WIDE",
    ) -> FramingDecision:
        """Analyzes video and produces scene-static crop windows for each scene.

        - If no cuts, 1 scene segment covering whole clip -> 1 crop window HOLD.
        - If cuts detected, new crop calculated at each scene cut and HELD.
        - NEVER continuous camera tracking.
        """
        if not os.path.exists(video_path):
            return FramingDecision(
                layout="REJECT",
                is_rejected=True,
                rejection_reason=f"Video file not found: {video_path}"
            )

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return FramingDecision(
                layout="REJECT",
                is_rejected=True,
                rejection_reason="Failed to open video stream"
            )

        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_duration = total_frames / fps if fps > 0 else 0.0

        if end_sec is None or end_sec <= start_sec:
            end_sec = video_duration

        clip_duration = max(0.1, end_sec - start_sec)

        # 1. Detect scene cuts if not passed
        if scene_cuts is None:
            scene_cuts = self.detect_scene_cuts(video_path, start_sec, end_sec)

        # Ensure cuts are sorted and within (0.2, clip_duration - 0.2)
        filtered_cuts = [
            c for c in sorted(scene_cuts)
            if 0.2 < c < (clip_duration - 0.2)
        ]

        # 2. Build scene intervals (clip-local timestamps)
        intervals: List[Tuple[float, float]] = []
        prev_t = 0.0
        for cut_t in filtered_cuts:
            if cut_t > prev_t:
                intervals.append((prev_t, cut_t))
                prev_t = cut_t
        intervals.append((prev_t, clip_duration))

        # 3. Process each scene segment with static crop
        crop_windows: List[SceneCrop] = []
        layouts_seen: List[str] = []
        speaker_modes: List[str] = []

        for seg_start, seg_end in intervals:
            seg_duration = seg_end - seg_start
            # Sample middle frame of segment
            sample_t = start_sec + seg_start + (seg_duration * 0.5)
            sample_frame_idx = int(sample_t * fps)

            cap.set(cv2.CAP_PROP_POS_FRAMES, sample_frame_idx)
            ret, frame = cap.read()
            faces = []
            if ret and frame is not None:
                faces = self.detect_faces(frame)

            crop, layout, spk_mode = self.calculate_scene_crop(
                faces=faces,
                frame_w=frame_w,
                frame_h=frame_h,
                scene_start=round(seg_start, 2),
                scene_end=round(seg_end, 2),
                allow_safe_full_frame=allow_safe_full_frame
            )
            crop_windows.append(crop)
            layouts_seen.append(layout)
            speaker_modes.append(spk_mode)

        cap.release()

        # 4. Resolve overall layout decision
        if "REJECT" in layouts_seen:
            return FramingDecision(
                layout="REJECT",
                crop_windows=crop_windows,
                speaker_mode=speaker_modes[0] if speaker_modes else "UNKNOWN",
                scene_cuts=filtered_cuts,
                is_rejected=True,
                rejection_reason="One or more scene segments were rejected (multi-speaker spread exceeds safe limit)"
            )

        if default_layout == "SAFE_WIDE":
            overall_layout = "SAFE_WIDE"
        else:
            overall_layout = "SAFE_FULL_FRAME" if "SAFE_FULL_FRAME" in layouts_seen else "PORTRAIT_9_16"
        dominant_speaker_mode = speaker_modes[0] if speaker_modes else "SINGLE_SPEAKER"

        return FramingDecision(
            layout=overall_layout,
            crop_windows=crop_windows,
            speaker_mode=dominant_speaker_mode,
            scene_cuts=filtered_cuts,
            is_rejected=False
        )
