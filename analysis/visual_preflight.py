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
from pydantic import BaseModel, Field

from llm_client import llm_client, DIRECT_VIDEO_VERIFIED

logger = logging.getLogger(__name__)


class VisualPreflightResult(BaseModel):
    """Result of Stage F Native Gemini Source-Clip Visual Preflight."""
    usable: bool = Field(..., description="True if source clip is visually viable for short-form")
    existing_visible_subtitles: bool = Field(
        default=False,
        description="True if video already contains burned-in subtitles on screen"
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
            "Kamu adalah Lead Visual Director untuk video vertical 9:16 (Shorts/TikTok/Reels).\n"
            "Tugasmu adalah menganalisis klip video mentah sumber sebelum diedit.\n"
            "Tugas terpentingmu:\n"
            "1. Cek apakah video sumber SUDAH MEMILIKI SUBTITLE/TEKS TERBAKAR di layar ('existing_visible_subtitles': true/false).\n"
            "2. Cek apakah subjek/pembicara terlihat jelas dan komposisinya layak ('subject_composition': 'acceptable'/'poor').\n"
            "3. Tentukan layout yang direkomendasikan ('recommended_layout': 'SAFE_WIDE' atau 'SAFE_ZOOM'). "
            "Ingat: 'SAFE_WIDE' adalah DEFAULT MUTLAK jika ada subtitle bawaan atau ragu!\n"
            "4. Berikan 2-3 observasi temporal dari titik berbeda dalam video untuk membuktikan video benar-benar ditonton.\n"
            "Format jawaban HANYA valid JSON."
        )

        prompt = (
            f"Tonton klip video ini dengan cermat dan evaluasi kelayakannya:\n\n"
            f"Transkrip dialog: \"{transcript_excerpt[:300]}\"\n\n"
            f"Kembalikan evaluasi HANYA dalam JSON valid:\n"
            f"{{\n"
            f'  "usable": true,\n'
            f'  "existing_visible_subtitles": false,\n'
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
                if blocking:
                    usable = False

                result = VisualPreflightResult(
                    usable=usable,
                    existing_visible_subtitles=bool(parsed.get("existing_visible_subtitles", False)),
                    shot_complexity=str(parsed.get("shot_complexity", "low")),
                    subject_composition=str(parsed.get("subject_composition", "acceptable")),
                    recommended_layout=str(parsed.get("recommended_layout", "SAFE_WIDE")),
                    blocking_issues=blocking,
                    notes=str(parsed.get("notes", "Evaluated by Gemini Visual Preflight")),
                    preflight_mode="GEMINI_NATIVE_VIDEO",
                    temporal_observations=list(parsed.get("temporal_observations", [])),
                )
                logger.info(f"Gemini Native Video Preflight SUCCESS: usable={result.usable}, "
                           f"subtitles={result.existing_visible_subtitles}, layout={result.recommended_layout}")
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
            existing_visible_subtitles=False,
            shot_complexity="low",
            subject_composition="acceptable",
            recommended_layout="SAFE_WIDE",
            blocking_issues=[],
            notes="Conservative deterministic fallback (SAFE_WIDE enforced, Gemini unavailable)",
            preflight_mode="LOCAL_FALLBACK",
        )
