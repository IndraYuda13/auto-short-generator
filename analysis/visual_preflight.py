"""Native Gemini Source-Clip Visual Preflight (Auto Clipper V3.1 Stage F).

Sends the raw candidate source MP4 directly to Gemini 3.8 Flash via 9router
to inspect the visual viability BEFORE any rendering occurs.

CRITICAL FIX: Pre-slices the candidate window from the full source video
before sending to Gemini. Full podcast files (100-200MB) cause HTTP 400.
Only the ~30-55 second candidate window (~2-8MB) is sent.

Records explicit preflight_mode: GEMINI_NATIVE_VIDEO | LOCAL_FALLBACK | FAILED
"""

import json
import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field, model_validator

from llm_client import llm_client, DIRECT_VIDEO_VERIFIED

logger = logging.getLogger(__name__)


class VisualPreflightResult(BaseModel):
    """Result of Stage F Native Gemini Source-Clip Visual Preflight."""
    usable: bool = Field(..., description="True if source clip is visually viable for short-form")
    has_subtitles: bool = Field(
        default=False,
        description="True if video already contains burned-in subtitles on screen"
    )
    existing_visible_subtitles: bool = Field(
        default=False,
        description="Backward-compatible alias for has_subtitles"
    )
    subtitle_confidence: float = Field(
        default=1.0,
        description="Confidence in subtitle detection (0.0 to 1.0)"
    )
    subtitle_reason: str = Field(
        default="",
        description="Reasoning/evidence for subtitle detection"
    )
    ending_complete: bool = Field(
        default=True,
        description="True if the sentence/thought at clip ending is complete, not cut off mid-sentence"
    )
    ending_natural: bool = Field(
        default=True,
        description="True if the ending feels natural and complete, not abrupt"
    )
    ending_reason: str = Field(
        default="",
        description="Reasoning/evidence for sentence ending evaluation"
    )
    shot_complexity: str = Field(
        default="low",
        description="low (single shot), medium (few cuts), high (rapid chaotic montage)"
    )
    subject_composition: str = Field(
        default="acceptable",
        description="acceptable or poor (subject missing/off-frame)"
    )
    recommended_layout: str = Field(
        default="SAFE_WIDE",
        description="SAFE_WIDE (default) or SAFE_ZOOM"
    )
    blocking_issues: List[str] = Field(
        default_factory=list,
        description="List of visual defects that preclude publication"
    )
    notes: str = Field(default="", description="Director observations")
    preflight_mode: str = Field(
        default="LOCAL_FALLBACK",
        description="GEMINI_NATIVE_VIDEO | LOCAL_FALLBACK | FAILED"
    )
    temporal_observations: List[Dict[str, str]] = Field(
        default_factory=list,
        description="Temporal observations proving video was consumed"
    )

    @model_validator(mode="before")
    @classmethod
    def _sync_fields(cls, data: Any) -> Any:
        if isinstance(data, dict):
            # Sync has_subtitles and existing_visible_subtitles
            if "has_subtitles" in data and "existing_visible_subtitles" not in data:
                data["existing_visible_subtitles"] = bool(data["has_subtitles"])
            elif "existing_visible_subtitles" in data and "has_subtitles" not in data:
                data["has_subtitles"] = bool(data["existing_visible_subtitles"])
            elif "has_subtitles" in data and "existing_visible_subtitles" in data:
                val = bool(data["has_subtitles"] or data["existing_visible_subtitles"])
                data["has_subtitles"] = val
                data["existing_visible_subtitles"] = val

            # Map confidence if present
            if "confidence" in data and "subtitle_confidence" not in data:
                data["subtitle_confidence"] = float(data["confidence"])

            # Map reason if present
            if "reason" in data and "subtitle_reason" not in data:
                data["subtitle_reason"] = str(data["reason"])
        return data


class VisualPreflight:
    """Preflights candidate video clips using direct video inspection via Gemini."""

    MAX_PREFLIGHT_FILE_SIZE_MB: float = 20.0  # Max file size to send to Gemini

    def __init__(self, client=None):
        self.client = client or llm_client

    def _pre_slice_candidate(
        self,
        video_path: str,
        start_sec: float = 0.0,
        duration_sec: float = 55.0,
    ) -> Optional[str]:
        """Pre-slices the candidate window from a full source video.

        Returns path to temporary sliced MP4, or None on failure.
        """
        if not os.path.exists(video_path):
            return None

        # If file is already small enough, use it directly
        file_size_mb = os.path.getsize(video_path) / (1024 * 1024)
        if file_size_mb <= self.MAX_PREFLIGHT_FILE_SIZE_MB:
            return video_path

        # Create temp slice
        tmp_path = tempfile.mktemp(suffix=".mp4", prefix="preflight_slice_")
        cmd = [
            "ffmpeg", "-y",
            "-ss", f"{start_sec:.3f}",
            "-i", video_path,
            "-t", f"{duration_sec:.3f}",
            "-c:v", "libx264", "-preset", "ultrafast", "-crf", "28",
            "-c:a", "aac", "-b:a", "64k",
            "-movflags", "+faststart",
            tmp_path,
        ]
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            if res.returncode == 0 and os.path.exists(tmp_path) and os.path.getsize(tmp_path) > 10000:
                return tmp_path
        except Exception as e:
            logger.warning(f"Failed to pre-slice candidate for preflight: {e}")

        return None

    def preflight_clip(
        self,
        video_path: Union[str, Path],
        transcript_excerpt: str = "",
        start_sec: float = 0.0,
        duration_sec: float = 55.0,
    ) -> VisualPreflightResult:
        """Inspects source candidate clip before rendering.

        Pre-slices the candidate window if the source file is too large for Gemini.
        """
        video_path = Path(video_path)
        if not video_path.exists():
            return VisualPreflightResult(
                usable=False,
                blocking_issues=[f"Video file not found: {video_path}"],
                notes="File missing on disk",
                preflight_mode="FAILED",
            )

        # Pre-slice to candidate window for Gemini
        slice_path = self._pre_slice_candidate(
            str(video_path), start_sec=start_sec, duration_sec=duration_sec
        )
        inspect_path = Path(slice_path) if slice_path else video_path
        is_temp_slice = (slice_path is not None and slice_path != str(video_path))

        system_prompt = (
            "Kamu adalah Lead Visual Director & Audio-Visual Inspector untuk video vertical 9:16 (Shorts/TikTok/Reels).\n"
            "Tugasmu adalah menganalisis klip video mentah sumber sebelum diedit.\n"
            "Tugas terpentingmu:\n"
            "1. Cek apakah video sumber SUDAH MEMILIKI SUBTITLE/TEKS TERBAKAR di layar:\n"
            "   - 'has_subtitles': true/false (true jika ada teks subtitle/caption yang menempel di video)\n"
            "   - 'confidence': float 0.0 sampai 1.0 (tingkat keyakinan deteksi subtitle)\n"
            "   - 'reason': penjelasan deteksi subtitle (misal ada teks subtitle di area bawah/tengah layar)\n"
            "2. Cek apakah AKHIR KLIP (ENDING) MENYELESAIKAN KALIMAT SECARA TUNTAS ATAU TERPOTONG:\n"
            "   - 'ending_complete': true/false (false jika pembicara terpotong di tengah kalimat seperti 'waktu itu masih...', 'jadi hidup...', 'karena sebenarnya...')\n"
            "   - 'ending_natural': true/false (false jika akhir klip terasa menggantung, patah, atau terpotong abrupt)\n"
            "   - 'ending_reason': penjelasan apakah kalimat penutup tuntas secara semantik dan akustik\n"
            "   Catatan: Jeda alami (tawa, tarikan napas, hening wajar) di tengah dialog BUKAN cacat; pastikan kalimat terakhir tuntas.\n"
            "3. Cek apakah subjek/pembicara terlihat jelas dan komposisinya layak ('subject_composition': 'acceptable'/'poor').\n"
            "4. Tentukan layout yang direkomendasikan ('recommended_layout': 'SAFE_WIDE' atau 'SAFE_ZOOM'). "
            "Ingat: 'SAFE_WIDE' adalah DEFAULT MUTLAK jika ada subtitle bawaan atau ragu!\n"
            "5. Berikan 2-3 observasi temporal dari titik berbeda dalam video untuk membuktikan video benar-benar ditonton.\n"
            "Format jawaban HANYA valid JSON."
        )

        prompt = (
            f"Tonton klip video ini dengan cermat dan evaluasi kelayakannya:\n\n"
            f"Transkrip dialog: \"{transcript_excerpt[:300]}\"\n\n"
            f"Kembalikan evaluasi HANYA dalam JSON valid:\n"
            f"{{\n"
            f'  "usable": true,\n'
            f'  "has_subtitles": false,\n'
            f'  "confidence": 0.95,\n'
            f'  "reason": "Tidak ada subtitle bawaan pada layar.",\n'
            f'  "ending_complete": true,\n'
            f'  "ending_natural": true,\n'
            f'  "ending_reason": "Kalimat penutup selesai dengan tuntas.",\n'
            f'  "shot_complexity": "low",\n'
            f'  "subject_composition": "acceptable",\n'
            f'  "recommended_layout": "SAFE_WIDE",\n'
            f'  "blocking_issues": [],\n'
            f'  "temporal_observations": [\n'
            f'    {{"approx_time": "00:05", "observation": "..."}},\n'
            f'    {{"approx_time": "00:20", "observation": "..."}}\n'
            f'  ],\n'
            f'  "notes": "Penjelasan."\n'
            f"}}"
        )

        try:
            raw = self.client.video_completion(
                video_path=inspect_path,
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=0.1,
            )
            parsed = self.client.extract_json(raw)
            if parsed:
                usable = bool(parsed.get("usable", True))
                blocking = list(parsed.get("blocking_issues", []))

                has_subs = bool(parsed.get("has_subtitles", parsed.get("existing_visible_subtitles", False)))
                sub_conf = float(parsed.get("confidence", parsed.get("subtitle_confidence", 1.0)))
                sub_reason = str(parsed.get("reason", parsed.get("subtitle_reason", "")))

                ending_comp = bool(parsed.get("ending_complete", True))
                ending_nat = bool(parsed.get("ending_natural", True))
                ending_reason = str(parsed.get("ending_reason", ""))

                if not ending_comp:
                    blocking.append(f"Incomplete sentence ending: {ending_reason or 'cut off mid-sentence'}")
                if not ending_nat:
                    blocking.append(f"Unnatural clip ending: {ending_reason or 'abrupt ending'}")

                if blocking:
                    usable = False

                result = VisualPreflightResult(
                    usable=usable,
                    has_subtitles=has_subs,
                    existing_visible_subtitles=has_subs,
                    subtitle_confidence=sub_conf,
                    subtitle_reason=sub_reason,
                    ending_complete=ending_comp,
                    ending_natural=ending_nat,
                    ending_reason=ending_reason,
                    shot_complexity=str(parsed.get("shot_complexity", "low")),
                    subject_composition=str(parsed.get("subject_composition", "acceptable")),
                    recommended_layout=str(parsed.get("recommended_layout", "SAFE_WIDE")),
                    blocking_issues=blocking,
                    notes=str(parsed.get("notes", "Evaluated by Gemini Visual Preflight")),
                    preflight_mode="GEMINI_NATIVE_VIDEO",
                    temporal_observations=list(parsed.get("temporal_observations", [])),
                )
                logger.info(
                    f"Gemini Native Video Preflight SUCCESS: usable={result.usable}, "
                    f"subtitles={result.has_subtitles} (conf={result.subtitle_confidence}), "
                    f"ending_complete={result.ending_complete}, layout={result.recommended_layout}"
                )
                return result

        except Exception as e:
            logger.warning(f"VisualPreflight invocation failed ({e}), using conservative fallback")

        finally:
            # Clean up temp slice
            if is_temp_slice and slice_path and os.path.exists(slice_path):
                try:
                    os.unlink(slice_path)
                except Exception:
                    pass

        # Fallback — DO NOT assume subtitles exist
        return VisualPreflightResult(
            usable=True,
            has_subtitles=False,
            existing_visible_subtitles=False,
            subtitle_confidence=1.0,
            subtitle_reason="Conservative deterministic fallback (SAFE_WIDE enforced, Gemini unavailable)",
            ending_complete=True,
            ending_natural=True,
            ending_reason="Deterministic fallback — assuming complete ending",
            shot_complexity="low",
            subject_composition="acceptable",
            recommended_layout="SAFE_WIDE",
            blocking_issues=[],
            notes="Conservative deterministic fallback (SAFE_WIDE enforced, Gemini unavailable)",
            preflight_mode="LOCAL_FALLBACK",
        )
