"""Analyzer module to find viral hooks using Gemini 3.8 Flash via 9router."""

import json
import logging
import re
from typing import List, Dict, Any, Optional

from config import settings
from llm_client import llm_client

logger = logging.getLogger(__name__)


class Analyzer:
    def analyze_transcript(
        self,
        video_title: str,
        transcript_segments: List[Dict[str, Any]],
        num_clips: int = 1
    ) -> List[Dict[str, Any]]:
        """
        Send transcript chunks to Gemini 3.8 Flash to detect viral hooks.
        Returns a list of clip candidates:
        [
            {
                "start_sec": float,
                "end_sec": float,
                "duration": float,
                "hook_score": int (1-100),
                "hook_reason": str,
                "title_clickbait": str,
                "description": str,
                "hashtags": str,
                "key_dialogue": str
            }
        ]
        """
        logger.info(f"Analyzing transcript ({len(transcript_segments)} segments) for viral hooks...")

        # Build compressed timestamped text representation
        lines = []
        for s in transcript_segments:
            start_m, start_s = divmod(int(s['start']), 60)
            timestamp = f"{start_m:02d}:{start_s:02d}"
            lines.append(f"[{timestamp} | {s['start']:.1f}s] {s['text']}")
        transcript_text = "\n".join(lines)

        # If transcript is excessively long (> 80k characters), truncate or take the most active parts
        if len(transcript_text) > 80000:
            transcript_text = transcript_text[:80000] + "\n...[TRUNCATED]..."

        system_prompt = (
            "Kamu adalah Senior Video Editor dan Growth Strategist spesialis konten short-form "
            "(YouTube Shorts & TikTok Reels) dengan pengalaman memproduksi miliaran views. "
            "Keahlian utamamu adalah menemukan momen perbincangan (dialog/monolog) yang memiliki "
            "Golden Hook (3 detik pertama menghentikan scroll), emosi tinggi, insight kontroversial/mengejutkan, "
            "atau punchline lucu, dan memiliki konteks lengkap yang memuaskan penonton tanpa terpotong aneh."
        )

        prompt = f"""Judul Video Sumber: "{video_title}"

Berikut adalah transkrip lengkap dengan timestamp (menit:detik | detik):
---
{transcript_text}
---

Tugasmu:
Pilih {num_clips} segmen klip terbaik untuk dijadikan YouTube Shorts / TikTok!
Syarat ketat durasi:
- Durasi minimal: {settings.MIN_CLIP_DURATION_SEC} detik.
- Durasi maksimal: {settings.MAX_CLIP_DURATION_SEC} detik.
- Nilai `start_sec` dan `end_sec` harus berupa angka float/integer dalam detik nyata sesuai timestamp transkrip di atas.
- Pastikan kalimat di awal segmen merupakan HOOK yang langsung menarik perhatian.
- Pastikan kalimat di akhir segmen tuntas (tidak menggantung di tengah kata).

Kembalikan jawaban HANYA dalam format JSON valid (array of objects):
[
  {{
    "start_sec": 124.5,
    "end_sec": 169.0,
    "hook_score": 95,
    "hook_reason": "Alasan hook ini sangat viral dalam 3 detik pertama",
    "title_clickbait": "Judul Clickbait Memancing Penasaran Tanpa Menipu (Maksimal 60 karakter)",
    "description": "Deskripsi singkat klip 1-2 kalimat untuk YouTube Shorts / TikTok",
    "hashtags": "#shorts #podcast #viral #indonesia #fyp",
    "key_dialogue": "Kutipan hook kata-kata pembuka segmen"
  }}
]"""

        try:
            raw_response = llm_client.chat_completion(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=0.2,
                json_mode=False
            )
            parsed = self._extract_json(raw_response)
            valid_clips = []
            for item in parsed:
                start = float(item.get("start_sec", 0))
                end = float(item.get("end_sec", 0))
                duration = end - start

                # Durability check: clamp or validate duration
                if duration < 15:
                    logger.warning(f"Clip too short ({duration}s), skipping.")
                    continue
                if duration > 60:
                    end = start + 55.0
                    duration = 55.0

                valid_clips.append({
                    "start_sec": start,
                    "end_sec": end,
                    "duration": duration,
                    "hook_score": int(item.get("hook_score", 85)),
                    "hook_reason": str(item.get("hook_reason", "")),
                    "title_clickbait": str(item.get("title_clickbait", video_title[:50])),
                    "description": str(item.get("description", "")),
                    "hashtags": str(item.get("hashtags", "#shorts #fyp")),
                    "key_dialogue": str(item.get("key_dialogue", ""))
                })

            logger.info(f"Analyzer found {len(valid_clips)} valid clips")
            return valid_clips

        except Exception as e:
            logger.error(f"Failed to analyze transcript with LLM: {e}")
            raise

    def _extract_json(self, text: str) -> List[Dict[str, Any]]:
        # Clean markdown codeblocks if present
        clean = re.sub(r"^```(?:json)?", "", text.strip(), flags=re.MULTILINE)
        clean = re.sub(r"```$", "", clean.strip(), flags=re.MULTILINE).strip()
        
        # Try direct parse
        try:
            data = json.loads(clean)
            if isinstance(data, list):
                return data
            if isinstance(data, dict):
                # Check if wrapped in clips key
                for k in ["clips", "data", "results"]:
                    if k in data and isinstance(data[k], list):
                        return data[k]
                return [data]
        except Exception:
            pass

        # Regex fallback to find [ { ... } ]
        match = re.search(r"\[.*\]", clean, flags=re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception:
                pass

        raise ValueError(f"Could not parse valid JSON from LLM response:\n{text[:300]}...")


analyzer = Analyzer()
