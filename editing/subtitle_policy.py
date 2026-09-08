"""Subtitle Policy Module (Blueprint Bab 13).

Implements:
1. Classification:
   - NONE: Generate Subtitle V2 (ASS format, clean 2-5 words per phrase, 1-2 lines,
     white font + dark outline, bottom safe-zone).
   - EMBEDDED_TRACK: Extract built-in subtitle track directly.
   - BURNED_IN: JANGAN OCR dan JANGAN buat composite aneh!
     Gunakan safe full-frame yang menjaga lebar frame original.
     Jika tidak muat atau merusak komposisi: tolak (reject candidate).
"""

import os
import re
import json
import logging
import subprocess
from enum import Enum
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import cv2
import numpy as np
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class SubtitleSourceType(str, Enum):
    NONE = "NONE"
    EMBEDDED_TRACK = "EMBEDDED_TRACK"
    BURNED_IN = "BURNED_IN"


class SubtitleAction(str, Enum):
    GENERATE = "GENERATE"
    EXTRACT_EMBEDDED = "EXTRACT_EMBEDDED"
    PRESERVE_BURNED_IN = "PRESERVE_BURNED_IN"
    REJECT = "REJECT"


class SubtitleClassificationResult(BaseModel):
    """Result of subtitle source inspection and policy selection."""
    source_type: SubtitleSourceType
    action: SubtitleAction
    recommended_layout: str = Field(
        default="PORTRAIT_9_16",
        description="Recommended layout: 'PORTRAIT_9_16', 'SAFE_FULL_FRAME', or 'REJECT'"
    )
    embedded_stream_index: Optional[int] = None
    burned_in_region: Optional[Dict[str, float]] = None
    is_rejected: bool = False
    rejection_reason: Optional[str] = None


def escape_ass_text(text: str) -> str:
    """Escapes backslashes and curly braces for ASS subtitle format."""
    text = text.replace("\\", "\\\\")
    text = text.replace("{", "\\{").replace("}", "\\}")
    return text.strip()


def format_ass_time(seconds: float) -> str:
    """Formats float seconds into ASS timestamp format: H:MM:SS.cs"""
    seconds = max(0.0, float(seconds))
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    cs = int(round((seconds - int(seconds)) * 100))
    if cs >= 100:
        s += 1
        cs -= 100
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def chunk_words_to_phrases(
    words_or_segments: List[Dict[str, Any]],
    min_words: int = 2,
    max_words: int = 5
) -> List[Dict[str, Any]]:
    """Groups words into clean 2-5 words phrases for Subtitle V2.

    Accepts:
    - List of word tokens: [{"word": "Halo", "start": 0.0, "end": 0.4}, ...]
    - Or list of segments: [{"text": "Halo semua", "start": 0.0, "end": 1.2}, ...]
    """
    if not words_or_segments:
        return []

    # Check if input is word tokens or segments
    first_item = words_or_segments[0]
    is_word_level = "word" in first_item

    if not is_word_level:
        # Segment level input: break sentences into 2-5 word phrases
        phrases: List[Dict[str, Any]] = []
        for seg in words_or_segments:
            text = str(seg.get("text", "")).strip()
            if not text:
                continue
            seg_start = float(seg.get("start", 0.0))
            seg_end = float(seg.get("end", seg_start + 1.0))
            tokens = text.split()
            if len(tokens) <= max_words:
                phrases.append({
                    "text": text,
                    "start": seg_start,
                    "end": seg_end
                })
            else:
                # Subdivide tokens evenly
                num_chunks = int(np.ceil(len(tokens) / max_words))
                chunk_len = int(np.ceil(len(tokens) / num_chunks))
                chunk_duration = (seg_end - seg_start) / num_chunks

                for idx in range(num_chunks):
                    sub_tokens = tokens[idx * chunk_len : (idx + 1) * chunk_len]
                    if not sub_tokens:
                        continue
                    p_start = seg_start + (idx * chunk_duration)
                    p_end = seg_start + ((idx + 1) * chunk_duration)
                    phrases.append({
                        "text": " ".join(sub_tokens),
                        "start": round(p_start, 2),
                        "end": round(p_end, 2)
                    })
        return phrases

    # Word-level input: group 2-5 words respecting punctuation
    PUNCT_SPLIT = {".", "!", "?", ",", ";", ":"}
    chunks: List[Dict[str, Any]] = []
    current_tokens: List[Dict[str, Any]] = []

    for w in words_or_segments:
        current_tokens.append(w)
        word_text = str(w.get("word", "")).strip()
        ends_with_punct = any(word_text.endswith(p) for p in PUNCT_SPLIT)

        if len(current_tokens) >= max_words or (len(current_tokens) >= min_words and ends_with_punct):
            phrase_text = " ".join(str(t.get("word", "")).strip() for t in current_tokens)
            phrase_start = float(current_tokens[0].get("start", 0.0))
            phrase_end = float(current_tokens[-1].get("end", phrase_start + 0.5))
            chunks.append({
                "text": phrase_text,
                "start": round(phrase_start, 2),
                "end": round(phrase_end, 2)
            })
            current_tokens = []

    if current_tokens:
        phrase_text = " ".join(str(t.get("word", "")).strip() for t in current_tokens)
        phrase_start = float(current_tokens[0].get("start", 0.0))
        phrase_end = float(current_tokens[-1].get("end", phrase_start + 0.5))
        chunks.append({
            "text": phrase_text,
            "start": round(phrase_start, 2),
            "end": round(phrase_end, 2)
        })

    return chunks


def generate_clean_ass_subtitles(
    phrases_or_words: List[Dict[str, Any]],
    output_path: Optional[str] = None,
    font_name: str = "Montserrat",
    font_size: int = 52,
    margin_v: int = 520
) -> str:
    """Generates clean Subtitle V2 in ASS format (Blueprint Bab 13).

    Specs:
    - 2-5 words per phrase
    - 1-2 lines
    - White font (&H00FFFFFF) + dark outline (&H00000000)
    - Bottom safe-zone (margin_v=520 in 1080x1920 canvas)
    """
    phrases = chunk_words_to_phrases(phrases_or_words, min_words=2, max_words=5)

    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{font_name},{font_size},&H00FFFFFF,&H000000FF,&H00000000,&H80000000,1,0,0,0,100,100,0,0,1,4,2,2,40,40,{margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

    dialogue_lines: List[str] = []
    for item in phrases:
        raw_text = escape_ass_text(str(item.get("text", "")))
        words = raw_text.split()
        # If phrase has 4-5 words, format into 1-2 balanced lines
        if len(words) >= 4:
            mid = len(words) // 2
            formatted_text = " ".join(words[:mid]) + "\\N" + " ".join(words[mid:])
        else:
            formatted_text = raw_text

        start_str = format_ass_time(float(item.get("start", 0.0)))
        end_str = format_ass_time(float(item.get("end", float(item.get("start", 0.0)) + 1.0)))

        line = f"Dialogue: 0,{start_str},{end_str},Default,,0,0,0,,{formatted_text}"
        dialogue_lines.append(line)

    ass_content = header + "\n".join(dialogue_lines) + "\n"

    if output_path:
        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        out_file.write_text(ass_content, encoding="utf-8")

    return ass_content


class SubtitlePolicyClassifier:
    """Classifies video subtitle sources and decides policy (Blueprint Bab 13)."""

    def __init__(self):
        pass

    def check_embedded_subtitles(self, video_path: str) -> Optional[int]:
        """Checks if video container has embedded subtitle tracks via ffprobe.

        Returns subtitle stream index if found, else None.
        """
        if not os.path.exists(video_path):
            return None

        cmd = [
            "ffprobe",
            "-v", "error",
            "-select_streams", "s",
            "-show_entries", "stream=index,codec_name",
            "-of", "json",
            video_path
        ]
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            if res.returncode == 0:
                data = json.loads(res.stdout or "{}")
                streams = data.get("streams", [])
                if streams:
                    return int(streams[0].get("index", 0))
        except Exception as e:
            logger.debug(f"ffprobe subtitle check error: {e}")

        return None

    def check_burned_in_subtitles(
        self,
        video_path: str,
        start_sec: float = 0.0,
        end_sec: Optional[float] = None,
        sample_count: int = 15
    ) -> Tuple[bool, Optional[Dict[str, float]]]:
        """Detects presence of burned-in subtitles in bottom 30% of video frames.

        Uses Canny edge density & horizontal morphology (Blueprint Bab 13).
        """
        if not os.path.exists(video_path):
            return False, None

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return False, None

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_dur = total_frames / fps if fps > 0 else 0.0

        if end_sec is None or end_sec <= start_sec:
            end_sec = video_dur

        clip_dur = max(0.1, end_sec - start_sec)
        frame_indices = [
            int((start_sec + (i * clip_dur / max(1, sample_count))) * fps)
            for i in range(sample_count)
        ]

        burned_sub_votes = 0
        min_x1, min_y1, max_x2, max_y2 = 1.0, 1.0, 0.0, 0.0
        valid_frames = 0

        for f_idx in frame_indices:
            if f_idx >= total_frames:
                continue
            cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
            ret, frame = cap.read()
            if not ret or frame is None:
                continue

            valid_frames += 1
            h, w = frame.shape[:2]
            y_start = int(h * 0.70)
            roi = frame[y_start:, :]
            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray_roi, 80, 200)
            density = float(np.sum(edges > 0)) / float(edges.size)

            # Subtitle text edge density characteristic
            if 0.02 <= density <= 0.22:
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
                closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
                col_proj = np.sum(closed > 0, axis=0)
                active_cols = np.where(col_proj > (roi.shape[0] * 0.1))[0]
                if len(active_cols) > int(w * 0.15):
                    burned_sub_votes += 1
                    min_x1 = min(min_x1, active_cols[0] / float(w))
                    max_x2 = max(max_x2, active_cols[-1] / float(w))
                    min_y1 = min(min_y1, 0.70)
                    max_y2 = max(max_y2, 0.96)

        cap.release()

        if valid_frames == 0:
            return False, None

        has_burned = (burned_sub_votes / valid_frames) >= 0.30
        region = None
        if has_burned:
            region = {
                "x1": round(max(0.0, min_x1), 3),
                "y1": round(min_y1, 3),
                "x2": round(min(1.0, max_x2), 3),
                "y2": round(max_y2, 3),
            }

        return has_burned, region

    def classify(
        self,
        video_path: str,
        start_sec: float = 0.0,
        end_sec: Optional[float] = None,
        allow_safe_full_frame: bool = True
    ) -> SubtitleClassificationResult:
        """Classifies video subtitle state and resolves policy action.

        Priority order:
        1. EMBEDDED_TRACK: If container has subtitle stream -> extract it.
        2. BURNED_IN: If bottom 30% has burned subtitles:
           - JANGAN OCR dan JANGAN buat composite aneh!
           - Gunakan safe full-frame yang menjaga lebar frame original.
           - Jika safe full-frame ditolak: reject candidate.
        3. NONE: Generate Subtitle V2.
        """
        # 1. Check embedded tracks
        embedded_idx = self.check_embedded_subtitles(video_path)
        if embedded_idx is not None:
            return SubtitleClassificationResult(
                source_type=SubtitleSourceType.EMBEDDED_TRACK,
                action=SubtitleAction.EXTRACT_EMBEDDED,
                recommended_layout="PORTRAIT_9_16",
                embedded_stream_index=embedded_idx,
                is_rejected=False
            )

        # 2. Check burned-in subtitles
        has_burned, burned_region = self.check_burned_in_subtitles(
            video_path=video_path,
            start_sec=start_sec,
            end_sec=end_sec
        )
        if has_burned:
            if allow_safe_full_frame:
                # Safe full-frame preserves 100% original width (burned subs uncropped)
                return SubtitleClassificationResult(
                    source_type=SubtitleSourceType.BURNED_IN,
                    action=SubtitleAction.PRESERVE_BURNED_IN,
                    recommended_layout="SAFE_FULL_FRAME",
                    burned_in_region=burned_region,
                    is_rejected=False
                )
            else:
                # Rejection required
                return SubtitleClassificationResult(
                    source_type=SubtitleSourceType.BURNED_IN,
                    action=SubtitleAction.REJECT,
                    recommended_layout="REJECT",
                    burned_in_region=burned_region,
                    is_rejected=True,
                    rejection_reason="Burned-in subtitles present and safe full-frame layout is not permitted for candidate."
                )

        # 3. NONE -> Generate Subtitle V2
        return SubtitleClassificationResult(
            source_type=SubtitleSourceType.NONE,
            action=SubtitleAction.GENERATE,
            recommended_layout="PORTRAIT_9_16",
            is_rejected=False
        )

    @staticmethod
    def extract_embedded_subtitles(video_path: str, output_path: str, stream_idx: int = 0) -> bool:
        """Extracts embedded subtitle stream to an output file (.srt / .ass)."""
        cmd = [
            "ffmpeg",
            "-y",
            "-i", video_path,
            "-map", f"0:s:{stream_idx}",
            output_path
        ]
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
            return res.returncode == 0 and os.path.exists(output_path)
        except Exception as e:
            logger.error(f"Failed to extract embedded subtitles: {e}")
            return False
