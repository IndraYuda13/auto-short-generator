"""Gemini Multimodal Visual Director via 9router.

Connects to 9router proxy (port 20128) using OpenAI-compatible multimodal chat completions.
Extracts an efficient VisualContextPack (sampled keyframes, scene-cut frames, subtitle regions).
Produces validated Pydantic VisualDirectorResult.
Fails safely to deterministic local fallback on timeout, connection failure, or malformed response.
"""

import os
import json
import base64
import logging
from enum import Enum
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
import requests
import cv2
import numpy as np
from pydantic import BaseModel, Field

from config import settings
from subtitle_detector import SubtitleSource, SubtitleRegion

logger = logging.getLogger(__name__)


class ShotType(str, Enum):
    SINGLE_SPEAKER_CLOSEUP = "single_speaker_closeup"
    SINGLE_SPEAKER_MEDIUM = "single_speaker_medium"
    TWO_PERSON_WIDE = "two_person_wide"
    MULTI_PERSON_PANEL = "multi_person_panel"
    BROLL_OR_GRAPHIC = "broll_or_graphic"
    UNKNOWN = "unknown"


class RecommendedFraming(str, Enum):
    FACE_TRACKED = "FACE_TRACKED"
    SUBTITLE_SAFE_FULL_WIDTH = "SUBTITLE_SAFE_FULL_WIDTH"
    SUBTITLE_PRESERVE_COMPOSITE = "SUBTITLE_PRESERVE_COMPOSITE"
    BLURRED_FALLBACK = "BLURRED_FALLBACK"


class ContinuityReview(BaseModel):
    continuity_risk: str = Field(default="low", description="low | medium | high")
    empty_subject_frames_detected: bool = Field(default=False, description="Whether empty/subject-less frames detected")
    recommended_hold_previous_framing: bool = Field(default=False, description="Whether temporal hold is recommended")
    reasons: List[str] = Field(default_factory=list, description="Reasoning behind continuity assessment")


class VisualDirectorResult(BaseModel):
    has_existing_subtitle: bool = Field(..., description="Whether source video already has subtitles")
    subtitle_kind: SubtitleSource = Field(default=SubtitleSource.NONE, description="NONE | EMBEDDED_TRACK | BURNED_IN")
    subtitle_region: Optional[SubtitleRegion] = Field(default=None, description="Protected subtitle bounding box")
    shot_type: ShotType = Field(default=ShotType.UNKNOWN, description="Semantic classification of shot type")
    faces_visible: int = Field(default=1, ge=0, description="Estimated count of visible faces/speakers")
    recommended_framing: RecommendedFraming = Field(
        default=RecommendedFraming.FACE_TRACKED,
        description="Recommended layout: FACE_TRACKED | SUBTITLE_SAFE_FULL_WIDTH | SUBTITLE_PRESERVE_COMPOSITE | BLURRED_FALLBACK"
    )
    confidence: float = Field(default=0.9, ge=0.0, le=1.0, description="Confidence score")
    reasons: List[str] = Field(default_factory=list, description="Reasoning behind visual direction decisions")
    continuity_review: Optional[ContinuityReview] = Field(default=None, description="Semantic continuity and subject presence check")
    raw_response: Optional[str] = Field(default=None, description="Raw model response for audit")


class VisualContextPack(BaseModel):
    clip_id: str
    start_sec: float
    end_sec: float
    duration: float
    keyframes_base64: List[str] = Field(default_factory=list, description="Base64 JPEG keyframes")
    scene_cuts: List[float] = Field(default_factory=list)
    local_face_count: int = 1
    local_has_subtitle: bool = False
    transcript_excerpt: str = ""


class VisualDirector:
    """Multimodal semantic visual director using Gemini 3.8 Flash via 9router."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        timeout_sec: int = 45
    ):
        self.base_url = (base_url or settings.ROUTER_BASE_URL).rstrip("/")
        self.api_key = api_key or settings.ROUTER_API_KEY
        self.model = model or settings.LLM_MODEL
        self.timeout_sec = timeout_sec

    def extract_visual_context_pack(
        self,
        video_path: str,
        start_sec: float,
        end_sec: float,
        scene_cuts: Optional[List[float]] = None,
        transcript_excerpt: str = "",
        max_frames: int = 6
    ) -> VisualContextPack:
        """
        Samples 4-6 representative frames across the clip window:
        - Downscales to lightweight 480x270 JPEG (JPEG quality 70) for fast inference
        """
        duration = max(0.1, end_sec - start_sec)
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return VisualContextPack(
                clip_id=Path(video_path).stem,
                start_sec=start_sec,
                end_sec=end_sec,
                duration=duration,
                scene_cuts=scene_cuts or []
            )

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        sample_times = set()
        step = max(3.0, duration / float(max_frames))
        t = start_sec + 1.0
        while t < end_sec:
            sample_times.add(round(t, 2))
            t += step

        if scene_cuts:
            for sc in scene_cuts[:2]:
                cut_abs = start_sec + sc
                if start_sec <= cut_abs < end_sec:
                    sample_times.add(round(cut_abs + 0.2, 2))

        sorted_times = sorted(list(sample_times))[:max_frames]
        b64_frames = []

        for st in sorted_times:
            frame_idx = int(st * fps)
            if frame_idx >= total_frames:
                break
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret or frame is None:
                continue

            # Downscale frame for fast multimodal token transfer (480x270)
            small = cv2.resize(frame, (480, 270), interpolation=cv2.INTER_AREA)
            encode_ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 70])
            if encode_ok:
                b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
                b64_frames.append(b64)

        cap.release()

        return VisualContextPack(
            clip_id=Path(video_path).stem,
            start_sec=start_sec,
            end_sec=end_sec,
            duration=duration,
            keyframes_base64=b64_frames,
            scene_cuts=scene_cuts or [],
            transcript_excerpt=transcript_excerpt
        )

    def analyze(
        self,
        video_path: str,
        start_sec: float,
        end_sec: float,
        scene_cuts: Optional[List[float]] = None,
        transcript_excerpt: str = "",
        local_subtitle_result: Optional[Any] = None
    ) -> VisualDirectorResult:
        """
        Executes Visual Director analysis:
        1. Prepares VisualContextPack
        2. Sends multimodal frames to Gemini via 9router
        3. Parses and validates structured JSON response into VisualDirectorResult
        4. On any failure, falls back to deterministic local result
        """
        local_has_sub = False
        local_sub_kind = SubtitleSource.NONE
        local_region = None
        if local_subtitle_result is not None:
            local_has_sub = getattr(local_subtitle_result, "has_existing_subtitle", False)
            local_sub_kind = getattr(local_subtitle_result, "source", SubtitleSource.NONE)
            local_region = getattr(local_subtitle_result, "region", None)

        default_framing = (
            RecommendedFraming.SUBTITLE_PRESERVE_COMPOSITE
            if local_has_sub
            else RecommendedFraming.FACE_TRACKED
        )

        fallback_result = VisualDirectorResult(
            has_existing_subtitle=local_has_sub,
            subtitle_kind=local_sub_kind,
            subtitle_region=local_region,
            shot_type=ShotType.SINGLE_SPEAKER_MEDIUM,
            faces_visible=1,
            recommended_framing=default_framing,
            confidence=0.85,
            reasons=["Deterministic local fallback used (router not invoked or returned error)."],
            continuity_review=ContinuityReview(
                continuity_risk="low",
                empty_subject_frames_detected=False,
                recommended_hold_previous_framing=False,
                reasons=["Fallback heuristic evaluation"]
            )
        )

        try:
            pack = self.extract_visual_context_pack(
                video_path=video_path,
                start_sec=start_sec,
                end_sec=end_sec,
                scene_cuts=scene_cuts,
                transcript_excerpt=transcript_excerpt,
                max_frames=6
            )
        except Exception as e:
            logger.warning(f"Failed to extract visual context pack: {e}. Using fallback.")
            return fallback_result

        if not pack.keyframes_base64:
            logger.warning("No keyframes extracted for VisualContextPack. Using fallback.")
            return fallback_result

        system_prompt = (
            "You are the Senior Visual Director for an automated short-form video editing system. "
            "You evaluate video frames to classify: "
            "1. Whether the source video already contains subtitles/captions (burned-in or overlay). "
            "   If present, specify their normalized bounding box [x1, y1, x2, y2] (0.0 to 1.0). "
            "2. Shot type (single_speaker_closeup, single_speaker_medium, two_person_wide, multi_person_panel). "
            "3. Framing safety: If existing subtitles are detected, recommend SUBTITLE_PRESERVE_COMPOSITE "
            "   to preserve authentic pixel subtitles at the bottom while framing the speaker nicely in portrait. "
            "   If no subtitles exist and single speaker is clear, recommend FACE_TRACKED. "
            "4. Continuity review: Check if there is high continuity risk, empty subject frames, or if previous framing should be held. "
            "Return valid JSON strictly adhering to the schema."
        )

        user_content: List[Dict[str, Any]] = [
            {
                "type": "text",
                "text": (
                    f"Analyze these {len(pack.keyframes_base64)} representative frames from a video clip "
                    f"(duration {pack.duration:.1f}s).\n"
                    f"Transcript excerpt: '{pack.transcript_excerpt[:200]}'\n\n"
                    "Respond with a JSON object matching this schema:\n"
                    "{\n"
                    '  "has_existing_subtitle": true/false,\n'
                    '  "subtitle_kind": "NONE" | "BURNED_IN" | "EMBEDDED_TRACK",\n'
                    '  "subtitle_region": {"x1": float, "y1": float, "x2": float, "y2": float, "protected": true} or null,\n'
                    '  "shot_type": "single_speaker_closeup" | "single_speaker_medium" | "two_person_wide" | "multi_person_panel",\n'
                    '  "faces_visible": integer,\n'
                    '  "recommended_framing": "FACE_TRACKED" | "SUBTITLE_SAFE_FULL_WIDTH" | "BLURRED_FALLBACK",\n'
                    '  "confidence": float (0.0 - 1.0),\n'
                    '  "reasons": ["reason1", "reason2"]\n'
                    "}"
                )
            }
        ]

        for b64 in pack.keyframes_base64:
            user_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
            })

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            "temperature": 0.0,
            "stream": False
        }

        url = f"{self.base_url}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        try:
            logger.info(f"Invoking Gemini Visual Director via 9router ({len(pack.keyframes_base64)} frames)...")
            res = requests.post(url, headers=headers, json=payload, timeout=self.timeout_sec)
            res.raise_for_status()
            data = res.json()
            choices = data.get("choices", [])
            if not choices:
                raise ValueError("No choices in 9router response")
            raw_text = choices[0].get("message", {}).get("content", "")

            cleaned = raw_text.strip()
            if "```json" in cleaned:
                cleaned = cleaned.split("```json")[1].split("```")[0].strip()
            elif "```" in cleaned:
                cleaned = cleaned.split("```")[1].split("```")[0].strip()

            parsed = json.loads(cleaned)

            sub_region = None
            if parsed.get("subtitle_region"):
                reg = parsed["subtitle_region"]
                sub_region = SubtitleRegion(
                    x1=float(reg.get("x1", 0.05)),
                    y1=float(reg.get("y1", 0.70)),
                    x2=float(reg.get("x2", 0.95)),
                    y2=float(reg.get("y2", 0.95)),
                    protected=True
                )
            elif parsed.get("has_existing_subtitle"):
                sub_region = SubtitleRegion(x1=0.06, y1=0.72, x2=0.94, y2=0.96, protected=True)

            sub_kind = SubtitleSource(parsed.get("subtitle_kind", "NONE").upper())
            if parsed.get("has_existing_subtitle") and sub_kind == SubtitleSource.NONE:
                sub_kind = SubtitleSource.BURNED_IN

            rec_framing = RecommendedFraming(parsed.get("recommended_framing", "FACE_TRACKED"))
            if (sub_kind != SubtitleSource.NONE or parsed.get("has_existing_subtitle")) and rec_framing in (RecommendedFraming.FACE_TRACKED, RecommendedFraming.SUBTITLE_SAFE_FULL_WIDTH):
                rec_framing = RecommendedFraming.SUBTITLE_PRESERVE_COMPOSITE

            cont_data = parsed.get("continuity_review")
            cont_review = None
            if cont_data and isinstance(cont_data, dict):
                cont_review = ContinuityReview(
                    continuity_risk=str(cont_data.get("continuity_risk", "low")),
                    empty_subject_frames_detected=bool(cont_data.get("empty_subject_frames_detected", False)),
                    recommended_hold_previous_framing=bool(cont_data.get("recommended_hold_previous_framing", False)),
                    reasons=cont_data.get("reasons", [])
                )
            else:
                cont_review = ContinuityReview(
                    continuity_risk="low",
                    empty_subject_frames_detected=False,
                    recommended_hold_previous_framing=False,
                    reasons=["No continuity risks detected"]
                )

            vd_res = VisualDirectorResult(
                has_existing_subtitle=bool(parsed.get("has_existing_subtitle", False)),
                subtitle_kind=sub_kind,
                subtitle_region=sub_region,
                shot_type=ShotType(parsed.get("shot_type", "unknown")),
                faces_visible=int(parsed.get("faces_visible", 1)),
                recommended_framing=rec_framing,
                confidence=float(parsed.get("confidence", 0.9)),
                reasons=parsed.get("reasons", ["Structured Visual Director response"]),
                continuity_review=cont_review,
                raw_response=raw_text
            )
            logger.info(
                f"Visual Director result: sub={vd_res.has_existing_subtitle} ({vd_res.subtitle_kind.value}), "
                f"framing={vd_res.recommended_framing.value}, shot={vd_res.shot_type.value}"
            )
            return vd_res

        except Exception as e:
            logger.warning(f"Gemini Visual Director invocation/parsing failed: {e}. Falling back to deterministic local result.")
            return fallback_result


visual_director = VisualDirector()
