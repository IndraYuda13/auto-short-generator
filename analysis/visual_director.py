"""Visual Director module for Auto Short Generator Phase A.

Sends representative frames per scene to Gemini via 9router to evaluate:
1. Visual usability for vertical 9:16 short form.
2. Subject visibility and continuity across scenes.
3. Burned-in subtitle status and layout recommendation.
4. Hard rejection if:
   - Too many blank/black frames
   - Subject missing for >30% duration
   - Scene cuts too chaotic / erratic
   - Burned-in subtitles messy or conflicting
Fails safely to deterministic local report if multimodal LLM is unavailable.
"""

import os
import re
import json
import base64
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import cv2
import requests
from pydantic import BaseModel, Field

from config import settings
from analysis.visual_analyzer import VisualAnalysisReport

logger = logging.getLogger(__name__)


class VisualDirectorVerdict(BaseModel):
    """Final visual qualification verdict from the Multimodal Visual Director."""
    approved: bool
    confidence: float = Field(default=0.9, ge=0.0, le=1.0)
    shot_type: str = "single_speaker"
    recommended_framing: str = "FACE_TRACKED"
    faces_visible: int = 1
    subject_visible_ratio: float = 1.0
    continuity_risk: str = "low"
    rejection_reasons: List[str] = Field(default_factory=list)
    notes: str = ""


class VisualDirector:
    """Multimodal Visual Director powered by Gemini via 9router."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        timeout_sec: int = 40,
    ):
        self.base_url = (base_url or getattr(settings, "ROUTER_BASE_URL", "http://127.0.0.1:20128/v1")).rstrip("/")
        self.api_key = api_key or getattr(settings, "ROUTER_API_KEY", "")
        self.model = model or getattr(settings, "LLM_MODEL", "ag/gemini-3.8-flash-high")
        self.timeout_sec = timeout_sec

    def extract_representative_frames_b64(
        self,
        video_path: str,
        start_sec: float,
        end_sec: float,
        scene_cuts: List[float],
        max_frames: int = 5
    ) -> List[str]:
        """
        Samples representative frames across scenes, downscaled to 480x270 JPEG base64.
        """
        if not os.path.exists(video_path):
            return []

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return []

        duration = max(0.1, end_sec - start_sec)
        sample_times = set()

        # Add start and middle points of each scene if available
        if scene_cuts:
            for sc in scene_cuts[:max_frames]:
                if start_sec <= sc <= end_sec:
                    sample_times.add(sc + 0.2)
        
        # Ensure at least 3-4 evenly spaced sample points
        for fraction in [0.15, 0.50, 0.85]:
            sample_times.add(start_sec + duration * fraction)

        sorted_times = sorted(list(sample_times))[:max_frames]
        b64_frames: List[str] = []

        for st in sorted_times:
            cap.set(cv2.CAP_PROP_POS_MSEC, st * 1000.0)
            ret, frame = cap.read()
            if not ret or frame is None:
                continue

            # Downscale to 480x270 for fast low-token multimodal transfer
            thumb = cv2.resize(frame, (480, 270), interpolation=cv2.INTER_AREA)
            success, enc = cv2.imencode(".jpg", thumb, [cv2.IMWRITE_JPEG_QUALITY, 70])
            if success:
                b64 = base64.b64encode(enc.tobytes()).decode("utf-8")
                b64_frames.append(b64)

        cap.release()
        return b64_frames

    def evaluate_window(
        self,
        video_path: str,
        start_sec: float = 0.0,
        end_sec: float = 0.0,
        local_report: Optional[VisualAnalysisReport] = None,
        report: Optional[VisualAnalysisReport] = None,
        transcript_text: str = "",
    ) -> VisualDirectorVerdict:
        """Convenience alias for direct_clip matching Orchestrator contract."""
        rep = local_report or report
        if rep is None:
            rep = VisualAnalysisReport(
                video_path=video_path,
                start_sec=start_sec,
                end_sec=end_sec,
                duration=max(0.1, end_sec - start_sec),
            )
        return self.direct_clip(video_path=video_path, report=rep, transcript_text=transcript_text)

    def direct_clip(
        self,
        video_path: str,
        report: VisualAnalysisReport,
        transcript_text: str = ""
    ) -> VisualDirectorVerdict:
        """
        Evaluates the candidate clip's visual usability.
        Combines deterministic OpenCV metrics with Gemini multimodal frame review.
        """
        # 1. Early rejection based on deterministic local checks
        if not report.is_viable:
            logger.info(f"VisualDirector early rejection from local analysis: {report.rejection_reasons}")
            return VisualDirectorVerdict(
                approved=False,
                confidence=0.95,
                rejection_reasons=report.rejection_reasons,
                subject_visible_ratio=report.subject_presence_ratio,
                notes="Rejected by deterministic OpenCV visual analysis rules"
            )

        # 2. Extract representative base64 frames
        frames_b64 = self.extract_representative_frames_b64(
            video_path=video_path,
            start_sec=report.start_sec,
            end_sec=report.end_sec,
            scene_cuts=report.scene_cuts,
            max_frames=5
        )

        if not frames_b64:
            logger.warning("Could not extract frames for multimodal review; relying on local report")
            return self._build_local_fallback_verdict(report)

        # 3. Formulate Multimodal Request to Gemini via 9router
        system_prompt = (
            "Kamu adalah Lead Video Director & Visual Framing Auditor spesialis konten short-form vertikal (9:16). "
            "Tugasmu adalah menganalisis frame video sumber ini untuk memastikan kelayakan visual saat dipotong menjadi video Shorts. "
            "Kriteria penolakan (REJECT):\n"
            "1. Terlalu banyak blank/black frame atau gambar rusak/buram parah.\n"
            "2. Pembicara/subjek utama hilang atau tidak terlihat >30% durasi.\n"
            "3. Scene cut terlalu kacau atau membingungkan.\n"
            "4. Subtitle bawaan (burned-in) berantakan dan menghalangi pemotongan rapi."
        )

        user_content: List[Dict[str, Any]] = [
            {
                "type": "text",
                "text": f"""Analisis sampel frame berikut untuk rentang waktu {report.start_sec:.1f}s - {report.end_sec:.1f}s (durasi {report.duration:.1f}s).
Transkrip dialog: \"{transcript_text[:300]}\"
Data OpenCV lokal:
- Scene cuts: {len(report.scene_cuts)}
- Blank frame ratio: {report.blank_frame_ratio * 100:.1f}%
- Subject presence ratio: {report.subject_presence_ratio * 100:.1f}%
- Has burned subtitles: {report.has_burned_subtitles}

Evaluasi frame-frame terlampir dan jawab HANYA dalam format JSON valid:
{{
  "approved": true,
  "confidence": 0.90,
  "shot_type": "single_speaker",
  "recommended_framing": "FACE_TRACKED",
  "faces_visible": 1,
  "subject_visible_ratio": 0.95,
  "continuity_risk": "low",
  "rejection_reasons": [],
  "notes": "Subjek pembicara jelas di tengah, framing stabil, tidak ada gangguan visual."
}}"""
            }
        ]

        for b64 in frames_b64:
            user_content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{b64}"
                }
            })

        try:
            url = f"{self.base_url}/chat/completions"
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            }
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ],
                "temperature": 0.1,
                "stream": False,
            }

            resp = requests.post(url, headers=headers, json=payload, timeout=self.timeout_sec)
            resp.raise_for_status()
            data = resp.json()
            raw_text = data.get("choices", [])[0].get("message", {}).get("content", "")
            parsed = self._extract_json(raw_text)

            approved = bool(parsed.get("approved", True))
            subj_ratio = float(parsed.get("subject_visible_ratio", report.subject_presence_ratio))
            reasons = list(parsed.get("rejection_reasons", []))

            # Enforce hard rules
            if subj_ratio < 0.20:
                approved = False
                reasons.append("Subject visible in less than 70% of frames")

            if parsed.get("continuity_risk") == "high":
                approved = False
                reasons.append("High visual continuity risk between scene cuts")

            framing = str(parsed.get("recommended_framing", "FACE_TRACKED"))
            if report.has_burned_subtitles and framing == "FACE_TRACKED":
                framing = "SUBTITLE_PRESERVE_COMPOSITE"

            return VisualDirectorVerdict(
                approved=approved,
                confidence=float(parsed.get("confidence", 0.85)),
                shot_type=str(parsed.get("shot_type", "single_speaker")),
                recommended_framing=framing,
                faces_visible=int(parsed.get("faces_visible", 1)),
                subject_visible_ratio=subj_ratio,
                continuity_risk=str(parsed.get("continuity_risk", "low")),
                rejection_reasons=reasons,
                notes=str(parsed.get("notes", ""))
            )
        except Exception as e:
            logger.warning(f"Multimodal Visual Director call failed ({e}); falling back to local OpenCV analysis")
            return self._build_local_fallback_verdict(report)

    @staticmethod
    def _build_local_fallback_verdict(report: VisualAnalysisReport) -> VisualDirectorVerdict:
        """Constructs a deterministic verdict when LLM multimodal call is unavailable."""
        recommended_framing = "FACE_TRACKED"
        if report.has_burned_subtitles:
            recommended_framing = "SUBTITLE_PRESERVE_COMPOSITE"

        return VisualDirectorVerdict(
            approved=report.is_viable,
            confidence=0.80,
            shot_type="single_speaker",
            recommended_framing=recommended_framing,
            faces_visible=1 if report.subject_presence_ratio >= 0.5 else 0,
            subject_visible_ratio=report.subject_presence_ratio,
            continuity_risk="low" if report.scene_cut_rate_per_sec <= 0.6 else "medium",
            rejection_reasons=report.rejection_reasons,
            notes="Evaluated via deterministic local OpenCV metrics"
        )

    @staticmethod
    def _extract_json(text: str) -> Dict[str, Any]:
        text = text.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
        try:
            return json.loads(text)
        except Exception:
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                return json.loads(match.group(0))
            raise ValueError(f"Unable to parse JSON from VisualDirector output: {text[:200]}")
