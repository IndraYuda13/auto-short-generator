"""Perceptual Quality Control (QC) module powered by Gemini Visual Director via 9router.

Implements Blueprint Bab 16.3:
- Generates 3x3 contact sheet image from 9 sampled frames across the clip
- Sends contact sheet to Gemini via 9router (http://127.0.0.1:20128/v1) along with:
  * Transcript text
  * EditPlan contract
  * Technical QC & local visual QC facts
- Evaluates:
  * publishable (bool)
  * score (0-100)
  * blocking_issues (List[str])
  * notes (str)
- Strict policy:
  * If blocking_issues present: reject / skip (publishable = False, do NOT auto-upload)
  * Maximum 1 deterministic repair attempt if possible (never infinite loop)
- Exports typed Pydantic PerceptualQCResult data contract
"""

import base64
import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import cv2
import numpy as np
import requests
from pydantic import BaseModel, Field

from config import settings
from quality.technical_qc import TechnicalQCResult
from quality.visual_qc import VisualQCResult

logger = logging.getLogger(__name__)


class PerceptualQCResult(BaseModel):
    """Structured result of Perceptual Quality Control evaluation (Blueprint Bab 16.3)."""
    passed: bool = Field(..., description="True if score >= threshold and no blocking issues")
    publishable: bool = Field(..., description="True if approved for auto-upload")
    score: int = Field(..., ge=0, le=100, description="Director quality score (0 to 100)")
    blocking_issues: List[str] = Field(default_factory=list, description="List of blocking issues preventing publication")
    notes: str = Field(default="", description="Director notes, reasoning, and visual observations")
    repair_attempted: bool = Field(default=False, description="Whether a deterministic repair was attempted")
    repair_action: Optional[str] = Field(default=None, description="Description of deterministic repair action if applied")


class PerceptualQC:
    """Multimodal Perceptual QC Inspector utilizing Gemini Visual Director via 9router."""

    DEFAULT_MIN_PASSING_SCORE: int = 70
    MAX_REPAIR_ATTEMPTS: int = 1

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        timeout_sec: int = 40,
        min_score: int = DEFAULT_MIN_PASSING_SCORE,
    ):
        self.base_url = (base_url or getattr(settings, "ROUTER_BASE_URL", "http://127.0.0.1:20128/v1")).rstrip("/")
        self.api_key = api_key or getattr(settings, "ROUTER_API_KEY", "")
        self.model = model or getattr(settings, "LLM_MODEL", "gemini/gemini-3.8-flash")
        self.timeout_sec = timeout_sec
        self.min_score = int(min_score)

    def generate_contact_sheet(
        self,
        video_path: Union[str, Path],
        grid_size: Tuple[int, int] = (3, 3),
        cell_size: Tuple[int, int] = (240, 426),
    ) -> Tuple[Optional[np.ndarray], Optional[str]]:
        """Generates a 3x3 contact sheet image from 9 sampled frames across the clip.

        Returns:
            (contact_sheet_bgr_array, base64_jpeg_string)
        """
        path = Path(video_path)
        if not path.exists():
            logger.error(f"Cannot generate contact sheet: file not found: {path}")
            return None, None

        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            logger.error(f"Cannot open video for contact sheet: {path}")
            return None, None

        rows, cols = grid_size
        num_cells = rows * cols  # 9 frames
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if total_frames > 0 else 0.0

        if duration <= 0.0 or total_frames < num_cells:
            # Handle very short clip or single frame
            sample_times = [duration * (i + 0.5) / max(1, num_cells) for i in range(num_cells)]
        else:
            sample_times = [duration * (i + 0.5) / num_cells for i in range(num_cells)]

        cell_w, cell_h = cell_size
        cells: List[np.ndarray] = []

        for t in sample_times:
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
            ret, frame = cap.read()
            if not ret or frame is None:
                # Fallback blank cell if read fails
                blank = np.zeros((cell_h, cell_w, 3), dtype=np.uint8)
                cells.append(blank)
                continue

            thumb = cv2.resize(frame, (cell_w, cell_h), interpolation=cv2.INTER_LINEAR)

            # Draw small timestamp badge in bottom-left corner
            label = f"{t:.1f}s"
            cv2.rectangle(thumb, (4, cell_h - 22), (58, cell_h - 4), (0, 0, 0), -1)
            cv2.putText(
                thumb, label, (8, cell_h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA
            )
            cells.append(thumb)

        cap.release()

        # Stitch cells into grid
        grid_rows: List[np.ndarray] = []
        for r in range(rows):
            row_cells = cells[r * cols : (r + 1) * cols]
            grid_rows.append(np.hstack(row_cells))

        contact_sheet = np.vstack(grid_rows)

        # Encode to JPEG base64
        success, enc = cv2.imencode(".jpg", contact_sheet, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not success:
            return contact_sheet, None

        b64_str = base64.b64encode(enc.tobytes()).decode("utf-8")
        return contact_sheet, b64_str

    def evaluate(
        self,
        video_path: Union[str, Path],
        transcript_text: str = "",
        edit_plan: Optional[Any] = None,
        technical_qc: Optional[Union[TechnicalQCResult, Dict[str, Any]]] = None,
        visual_qc: Optional[Union[VisualQCResult, Dict[str, Any]]] = None,
        repair_attempt: int = 0,
        repair_handler: Optional[Callable[[List[str], Any], Tuple[bool, Optional[str], Optional[str]]]] = None,
    ) -> PerceptualQCResult:
        """Evaluates video quality using Gemini Visual Director via 9router.

        If blocking_issues are detected and repair_attempt < MAX_REPAIR_ATTEMPTS:
        Attempts exactly 1 deterministic repair if repair_handler is provided, then re-evaluates.
        """
        path = Path(video_path)

        # Generate contact sheet
        sheet_img, b64_sheet = self.generate_contact_sheet(path)
        if sheet_img is None or b64_sheet is None:
            return PerceptualQCResult(
                passed=False,
                publishable=False,
                score=0,
                blocking_issues=["Failed to generate 3x3 contact sheet from video"],
                notes="Video file could not be read or frame extraction failed.",
                repair_attempted=False,
            )

        # Call Gemini Visual Director via 9router
        raw_result, error_msg = self._call_visual_director(
            b64_sheet=b64_sheet,
            transcript_text=transcript_text,
            edit_plan=edit_plan,
            technical_qc=technical_qc,
            visual_qc=visual_qc,
        )

        if raw_result is None:
            # Deterministic fallback when 9router is unreachable or times out
            logger.warning(f"9router Visual Director unavailable ({error_msg}); using deterministic fallback")
            return self._build_deterministic_fallback(
                technical_qc=technical_qc,
                visual_qc=visual_qc,
                notes_prefix=f"Deterministic fallback ({error_msg}): ",
            )

        # Extract evaluated metrics
        publishable = bool(raw_result.get("publishable", False))
        score = int(raw_result.get("score", 0))
        blocking_issues = list(raw_result.get("blocking_issues", []))
        notes = str(raw_result.get("notes", ""))

        # Blueprint rule: if blocking_issues are present, reject/skip (publishable = False)
        if blocking_issues:
            publishable = False
            passed = False
        else:
            passed = (publishable and score >= self.min_score)

        # Blueprint rule: maximum 1 deterministic repair attempt if possible, don't infinite loop
        if (
            not passed
            and blocking_issues
            and repair_attempt < self.MAX_REPAIR_ATTEMPTS
            and repair_handler is not None
        ):
            logger.info(
                f"Blocking issues found: {blocking_issues}. Invoking deterministic repair handler "
                f"(attempt {repair_attempt + 1}/{self.MAX_REPAIR_ATTEMPTS})."
            )
            success, repaired_path, action_taken = repair_handler(blocking_issues, edit_plan)
            if success and repaired_path and os.path.exists(repaired_path):
                # Re-evaluate repaired output exactly once
                repaired_result = self.evaluate(
                    video_path=repaired_path,
                    transcript_text=transcript_text,
                    edit_plan=edit_plan,
                    technical_qc=technical_qc,
                    visual_qc=visual_qc,
                    repair_attempt=repair_attempt + 1,
                    repair_handler=None,  # Do not allow further recursive repairs
                )
                repaired_result.repair_attempted = True
                repaired_result.repair_action = action_taken or "Deterministic repair applied"
                return repaired_result

        return PerceptualQCResult(
            passed=passed,
            publishable=publishable,
            score=score,
            blocking_issues=blocking_issues,
            notes=notes,
            repair_attempted=(repair_attempt > 0),
        )

    def _call_visual_director(
        self,
        b64_sheet: str,
        transcript_text: str,
        edit_plan: Optional[Any],
        technical_qc: Optional[Union[TechnicalQCResult, Dict[str, Any]]],
        visual_qc: Optional[Union[VisualQCResult, Dict[str, Any]]],
    ) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """Invokes Gemini Visual Director multimodal endpoint via 9router."""
        system_prompt = (
            "Kamu adalah Lead Visual Director & Quality Assurance Officer spesialis video vertikal 9:16 "
            "(YouTube Shorts / TikTok / Reels).\n"
            "Tugasmu adalah menganalisis lembar kontak (3x3 contact sheet) video yang telah dirender bersama transkrip, "
            "rencana editing (EditPlan), dan fakta teknis QC.\n"
            "Kriteria Mutlak:\n"
            "1. Pembicara/subjek utama harus tampak jelas, framing stabil, dan tidak terpotong aneh.\n"
            "2. Subtitle harus terbaca dengan jelas, tidak tumpang tindih, dan berada dalam safe-zone.\n"
            "3. Jika ada cacat visual fatal (wajah terpotong batas, black screen freeze, subtitle rusak/overlapping, "
            "gambar pecah parah), daftarkan di 'blocking_issues' dan set 'publishable': false.\n"
            "4. Jika video layak upload, set 'publishable': true dan berikan score 70-100.\n"
            "Format jawaban HANYA valid JSON tanpa markdown framing tambahan."
        )

        tech_summary = "N/A"
        if technical_qc is not None:
            if isinstance(technical_qc, BaseModel):
                tech_summary = json.dumps(technical_qc.model_dump())
            elif isinstance(technical_qc, dict):
                tech_summary = json.dumps(technical_qc)

        vis_summary = "N/A"
        if visual_qc is not None:
            if isinstance(visual_qc, BaseModel):
                vis_summary = json.dumps(visual_qc.model_dump())
            elif isinstance(visual_qc, dict):
                vis_summary = json.dumps(visual_qc)

        plan_summary = "N/A"
        if edit_plan is not None:
            if hasattr(edit_plan, "model_dump"):
                plan_summary = json.dumps(edit_plan.model_dump())
            elif isinstance(edit_plan, dict):
                plan_summary = json.dumps(edit_plan)
            else:
                plan_summary = str(edit_plan)

        prompt_text = (
            f"Analisis 3x3 contact sheet terlampir (9 sampel frame sepanjang video 9:16):\n\n"
            f"Transkrip dialog: \"{transcript_text[:400]}\"\n"
            f"EditPlan: {plan_summary}\n"
            f"Fakta Technical QC: {tech_summary}\n"
            f"Fakta Visual QC: {vis_summary}\n\n"
            f"Kembalikan evaluasi HANYA dalam JSON valid dengan skema:\n"
            f"{{\n"
            f'  "publishable": true,\n'
            f'  "score": 85,\n'
            f'  "blocking_issues": [],\n'
            f'  "notes": "Penjelasan detail mengenai komposisi visual, framing subjek, dan keterbacaan subtitle."\n'
            f"}}"
        )

        user_content: List[Dict[str, Any]] = [
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{b64_sheet}"
                }
            },
            {
                "type": "text",
                "text": prompt_text
            }
        ]

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            "temperature": 0.2,
            "max_tokens": 1024,
            "stream": False,
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        url = f"{self.base_url}/chat/completions"

        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=self.timeout_sec)
            resp.raise_for_status()
            text_body = getattr(resp, "text", "") or ""
            content = ""
            if text_body.strip().startswith("data:"):
                parts = []
                for line in text_body.strip().split("\n"):
                    line = line.strip()
                    if line.startswith("data:"):
                        chunk_str = line[5:].strip()
                        if chunk_str and chunk_str != "[DONE]":
                            try:
                                c = json.loads(chunk_str)
                                delta = c.get("choices", [{}])[0].get("delta", {})
                                if "content" in delta and delta["content"]:
                                    parts.append(delta["content"])
                            except Exception:
                                pass
                content = "".join(parts)
            else:
                data = resp.json()
                content = data["choices"][0]["message"]["content"]

            parsed = self._extract_json(content)
            if parsed is not None:
                return parsed, None
            return None, f"Failed to parse valid JSON from LLM response: {content[:100]}"
        except requests.Timeout:
            return None, f"9router request timed out after {self.timeout_sec}s"
        except Exception as e:
            return None, str(e)

    def _extract_json(self, text: str) -> Optional[Dict[str, Any]]:
        """Robustly extracts JSON object from response string."""
        if not text:
            return None

        # Clean markdown codeblocks
        cleaned = text.strip()
        if "```json" in cleaned:
            cleaned = cleaned.split("```json", 1)[1].split("```", 1)[0].strip()
        elif "```" in cleaned:
            cleaned = cleaned.split("```", 1)[1].split("```", 1)[0].strip()

        # Try direct parse
        try:
            return json.loads(cleaned)
        except Exception:
            pass

        # Regex fallback for JSON object {...}
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception:
                pass

        return None

    def _build_deterministic_fallback(
        self,
        technical_qc: Optional[Union[TechnicalQCResult, Dict[str, Any]]],
        visual_qc: Optional[Union[VisualQCResult, Dict[str, Any]]],
        notes_prefix: str = "",
    ) -> PerceptualQCResult:
        """Constructs deterministic fallback result when multimodal LLM is unavailable."""
        blocking_issues: List[str] = []

        # Check technical QC facts
        tech_passed = True
        if technical_qc is not None:
            if isinstance(technical_qc, TechnicalQCResult):
                tech_passed = technical_qc.passed
                if not tech_passed:
                    blocking_issues.extend(technical_qc.errors)
            elif isinstance(technical_qc, dict):
                tech_passed = bool(technical_qc.get("passed", False))
                if not tech_passed:
                    blocking_issues.extend(technical_qc.get("errors", []))

        # Check visual QC facts
        vis_passed = True
        if visual_qc is not None:
            if isinstance(visual_qc, VisualQCResult):
                vis_passed = visual_qc.passed
                if not vis_passed:
                    blocking_issues.extend(visual_qc.errors)
            elif isinstance(visual_qc, dict):
                vis_passed = bool(visual_qc.get("passed", False))
                if not vis_passed:
                    blocking_issues.extend(visual_qc.get("errors", []))

        if blocking_issues:
            score = 35
            publishable = False
            passed = False
            notes = f"{notes_prefix}Rejected based on technical/visual QC failures."
        elif tech_passed and vis_passed:
            score = 80
            publishable = True
            passed = True
            notes = f"{notes_prefix}Approved: All deterministic technical and visual QC criteria passed."
        else:
            score = 50
            publishable = False
            passed = False
            blocking_issues.append("QC verification incomplete or uncertain")
            notes = f"{notes_prefix}Incomplete verification data."

        return PerceptualQCResult(
            passed=passed,
            publishable=publishable,
            score=score,
            blocking_issues=blocking_issues,
            notes=notes,
            repair_attempted=False,
        )


def evaluate_perceptual_qc(
    video_path: Union[str, Path],
    transcript_text: str = "",
    edit_plan: Optional[Any] = None,
    technical_qc: Optional[Union[TechnicalQCResult, Dict[str, Any]]] = None,
    visual_qc: Optional[Union[VisualQCResult, Dict[str, Any]]] = None,
    min_score: int = PerceptualQC.DEFAULT_MIN_PASSING_SCORE,
) -> PerceptualQCResult:
    """Convenience function to evaluate perceptual visual QC on a video file."""
    qc = PerceptualQC(min_score=min_score)
    return qc.evaluate(
        video_path=video_path,
        transcript_text=transcript_text,
        edit_plan=edit_plan,
        technical_qc=technical_qc,
        visual_qc=visual_qc,
    )
