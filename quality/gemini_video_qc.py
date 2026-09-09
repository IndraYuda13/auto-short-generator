"""Native Gemini Final Video QC (Auto Clipper V3.1 Stage J).

Sends the FINAL RENDERED VIDEO directly to Gemini 3.8 Flash via 9router
to inspect the actual rendered output before authorizing publication.

Verifies:
1. Comfort, stability, and watchability.
2. Subject visibility and framing sanity.
3. Subtitle timing, zero overlap, and silence clearance (no lingering).
4. Subtitle text accuracy (verbatim what is heard; flags obvious transcription errors).
5. Audio/visual coherence.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field

from llm_client import llm_client, DIRECT_VIDEO_VERIFIED

logger = logging.getLogger(__name__)


class GeminiVideoQCResult(BaseModel):
    """Structured result of Stage J Native Gemini Final Video QC."""
    passed: bool = Field(..., description="True if video is approved for publication")
    score: int = Field(..., ge=0, le=100, description="Overall quality score (0-100)")
    subtitle_timing: str = Field(default="PASS", description="PASS or FAIL")
    subtitle_overlap: str = Field(default="PASS", description="PASS or FAIL")
    subtitle_linger: str = Field(default="PASS", description="PASS or FAIL")
    subtitle_text_accuracy: str = Field(default="PASS", description="PASS or FAIL")
    obvious_transcription_errors: List[Union[Dict[str, Any], str]] = Field(
        default_factory=list,
        description="List of obvious transcription errors {shown, heard, approx_time}"
    )
    blocking_reasons: List[str] = Field(
        default_factory=list,
        description="List of blocking defects preventing publication"
    )
    summary: str = Field(default="", description="Director review and summary")


class GeminiNativeVideoQC:
    """Final Quality Control gate evaluator powered by Gemini direct video inspection."""

    MIN_PASSING_SCORE: int = 70

    def __init__(self, client=None, min_score: int = MIN_PASSING_SCORE):
        self.client = client or llm_client
        self.min_score = min_score

    def evaluate_video(
        self,
        video_path: Union[str, Path],
        transcript_text: str = "",
        edit_plan: Optional[Any] = None,
    ) -> GeminiVideoQCResult:
        """Inspects final rendered MP4 directly via Gemini 3.8 Flash."""
        video_path = Path(video_path)
        if not video_path.exists():
            return GeminiVideoQCResult(
                passed=False,
                score=0,
                subtitle_timing="FAIL",
                subtitle_overlap="FAIL",
                subtitle_linger="FAIL",
                subtitle_text_accuracy="FAIL",
                blocking_reasons=[f"Rendered video file not found: {video_path}"],
                summary="File missing on disk"
            )

        system_prompt = (
            "Kamu adalah Senior Quality Assurance Lead & Executive Producer untuk YouTube Shorts dan Instagram Reels.\n"
            "Tugasmu adalah MENONTON video vertikal 9:16 yang telah selesai dirender dan memutuskan apakah video ini "
            "100% LAYAK TAYANG (PUBLISH) atau HARUS DITOLAK (FAIL).\n\n"
            "Evaluasi spesifik Subtitle & Visual (V3.1 Acceptance Standard):\n"
            "1. SUBTITLE OVERLAP: Pastikan ZERO overlap antara event subtitle (hanya 1 caption lane aktif pada satu waktu).\n"
            "2. SUBTITLE TIMING & LINGER: Subtitle harus muncul saat kata diucapkan dan HILANG saat jeda bicara (clearance saat silent gap >300ms).\n"
            "3. SUBTITLE TEXT ACCURACY: Teks subtitle yang tampil di layar harus PERSIS sesuai ucapan audio (verbatim). "
            "Periksa secara obyektif apakah ada kata yang salah dengar atau typo antara audio yang terdengar dan teks yang tampil di layar. "
            "Jika dan HANYA JIKA ada kesalahan teks yang benar-benar tampil di layar, laporkan dalam 'obvious_transcription_errors'. "
            "Jika teks di layar sudah sesuai dengan ucapan audio, biarkan 'obvious_transcription_errors': [].\n"
            "4. SUBTITLE TERPOTONG / SAFE-ZONE: Teks tidak boleh terpotong di tepi kiri, kanan, atau tertutup UI bawah.\n"
            "5. FRAMING & STABILITAS: Wajah tidak terpotong kasar, tidak ada frame kosong tanpa subjek.\n\n"
            "Aturan Keputusan:\n"
            "- Jika ada obvious transcription error atau subtitle_text_accuracy == 'FAIL', maka passed = false.\n"
            "- Jika ada subtitle overlap atau subtitle_timing == 'FAIL', maka passed = false.\n"
            "- Jika score < 70 atau ada blocking reasons, maka passed = false.\n"
            "Format jawaban HANYA valid JSON."
        )

        plan_summary = "N/A"
        if edit_plan is not None:
            if hasattr(edit_plan, "model_dump_json"):
                plan_summary = edit_plan.model_dump_json()
            elif isinstance(edit_plan, dict):
                plan_summary = json.dumps(edit_plan)
            else:
                plan_summary = str(edit_plan)

        prompt = (
            f"Tonton video short 9:16 terlampir ini dari detik 0 sampai akhir:\n\n"
            f"Transkrip dialog referensi yang sah (ground truth ucapan audio):\n\"{transcript_text}\"\n\n"
            f"Rencana Edit: {plan_summary}\n\n"
            "Catatan tipografi: Huruf 'i' kecil pada font sans-serif memiliki titik di atas garis vertikal pendek dan jangan salah diidentifikasi sebagai 'l'.\n\n"
            f"Kembalikan keputusan evaluasi HANYA dalam JSON valid:\n"
            f"{{\n"
            f'  "passed": true,\n'
            f'  "score": 88,\n'
            f'  "subtitle_timing": "PASS",\n'
            f'  "subtitle_overlap": "PASS",\n'
            f'  "subtitle_linger": "PASS",\n'
            f'  "subtitle_text_accuracy": "PASS",\n'
            f'  "obvious_transcription_errors": [],\n'
            f'  "blocking_reasons": [],\n'
            f'  "summary": "Analisis visual, akurasi teks subtitle, timing, dan kelayakan tayang."\n'
            f"}}"
        )

        try:
            raw = self.client.video_completion(
                video_path=video_path,
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=0.1,
            )
            parsed = self.client.extract_json(raw)
            if parsed:
                sub_timing = str(parsed.get("subtitle_timing", "PASS")).upper()
                sub_overlap = str(parsed.get("subtitle_overlap", "PASS")).upper()
                sub_linger = str(parsed.get("subtitle_linger", "PASS")).upper()
                sub_acc = str(parsed.get("subtitle_text_accuracy", "PASS")).upper()
                obvious_errors = list(parsed.get("obvious_transcription_errors", []))
                blocking = list(parsed.get("blocking_reasons", []))
                score = int(parsed.get("score", 0))

                # Strict V3.1 Gate: Any obvious error or text inaccuracy causes FAIL
                real_errors = []
                for err in obvious_errors:
                    if isinstance(err, dict):
                        shown = str(err.get("shown", "")).strip()
                        heard = str(err.get("heard", "")).strip()
                        err_desc = str(err.get("description", err.get("error", ""))).strip()
                        if not shown and not heard and not err_desc:
                            continue
                        if (
                            shown.lower() in ("none", "tidak ada", "tidak ada kesalahan", "clean", "pass", "no", "n/a", "-")
                            or heard.lower() in ("none", "tidak ada", "tidak ada kesalahan", "clean", "pass", "no", "n/a", "-")
                            or err_desc.lower() in ("none", "tidak ada", "tidak ada kesalahan", "clean", "pass", "no", "n/a", "-")
                        ):
                            continue
                        if shown or heard:
                            desc = f"Obvious subtitle error: '{shown}' instead of heard '{heard}'"
                        else:
                            desc = f"Obvious subtitle error: {err_desc}"
                        real_errors.append(err)
                    elif isinstance(err, str):
                        clean_err = err.lower().strip()
                        if clean_err in ("none", "tidak ada", "tidak ada kesalahan", "clean", "pass", "", "no", "n/a", "-", "[]"):
                            continue
                        desc = f"Obvious subtitle error: {err}"
                        real_errors.append(err)
                    else:
                        desc = f"Obvious subtitle error: {str(err)}"
                        real_errors.append(str(err))
                    if desc not in blocking:
                        blocking.append(desc)

                if real_errors:
                    sub_acc = "FAIL"
                    obvious_errors = real_errors
                else:
                    obvious_errors = []

                if sub_acc == "FAIL" and "Subtitle text accuracy failed" not in blocking:
                    blocking.append("Subtitle text accuracy failed")
                if sub_overlap == "FAIL" and "Subtitle overlap detected" not in blocking:
                    blocking.append("Subtitle overlap detected")
                if sub_timing == "FAIL" and "Subtitle timing mismatch" not in blocking:
                    blocking.append("Subtitle timing mismatch")

                passed = bool(parsed.get("passed", True))
                if blocking or score < self.min_score or sub_acc == "FAIL" or sub_overlap == "FAIL" or sub_timing == "FAIL":
                    passed = False

                return GeminiVideoQCResult(
                    passed=passed,
                    score=score,
                    subtitle_timing=sub_timing,
                    subtitle_overlap=sub_overlap,
                    subtitle_linger=sub_linger,
                    subtitle_text_accuracy=sub_acc,
                    obvious_transcription_errors=obvious_errors,
                    blocking_reasons=blocking,
                    summary=str(parsed.get("summary", "Evaluated by Gemini Native Video QC V3.1")),
                )
        except Exception as e:
            logger.warning(f"GeminiNativeVideoQC invocation failed ({e}), using conservative fallback")

        # Fallback — DO NOT silently approve. Report FAILED mode.
        return GeminiVideoQCResult(
            passed=False,
            score=0,
            subtitle_timing="FAIL",
            subtitle_overlap="FAIL",
            subtitle_linger="FAIL",
            subtitle_text_accuracy="FAIL",
            obvious_transcription_errors=[],
            blocking_reasons=["Gemini Native Video QC unavailable — cannot verify publishability"],
            summary="FAILED: Gemini QC invocation failed, video not approved for upload",
        )
