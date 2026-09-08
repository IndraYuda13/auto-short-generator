"""Subtitle V2 module: Clean chunking, calm formatting, ASS escaping, and safe-zone positioning.

Enforces:
- 2-4 words natural chunking
- Stable words (no aggressive 108% scale pop on every single word)
- Semantic emphasis highlight for words identified in EditPlan
- Safe-zone positioning (MarginV=520, Y=1380px) to prevent bottom UI overlap
- Full ASS reserved character escaping
- Static phrase-level fallback when only segment-level timing exists
"""

import os
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
from edit_plan import EditPlan, SubtitleStyle

# ASS formatting helpers
def escape_ass_text(text: str) -> str:
    """Escapes curly brackets, backslashes, and ASS special characters."""
    text = text.replace("\\", "\\\\")
    text = text.replace("{", "\\{").replace("}", "\\}")
    return text


def format_ass_time(seconds: float) -> str:
    """Formats seconds into ASS timestamp: H:MM:SS.cs"""
    seconds = max(0.0, seconds)
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    cs = int(round((seconds - int(seconds)) * 100))
    if cs >= 100:
        s += 1
        cs -= 100
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


class SubtitleGeneratorV2:
    """Generates clean, readable Advanced SubStation Alpha (.ass) subtitle files."""

    def __init__(self, fonts_dir: Optional[Path] = None):
        self.fonts_dir = fonts_dir

    @staticmethod
    def chunk_indonesian_words(words: List[Dict[str, Any]], max_words: int = 4) -> List[List[Dict[str, Any]]]:
        """
        Groups words into natural Indonesian phrase chunks (2 to 4 words):
        - Binds negation with following predicate (nggak, tidak, belum, bukan, ga, gak)
        - Binds prepositions with following noun (di, ke, dari, pada, untuk, dengan)
        - Treats conjunctions (kalau, karena, bahwa, tapi, sehingga, waktu) as boundary openers
        - Treats punctuation marks (., !, ?, ,, :) as hard phrase boundaries
        """
        if not words:
            return []

        NEGATIONS = {"nggak", "ngga", "ga", "gak", "tidak", "belum", "bukan", "tak"}
        PREPOSITIONS = {"di", "ke", "dari", "pada", "untuk", "dengan", "buat", "bagi"}
        CONJUNCTIONS = {"kalau", "karena", "bahwa", "tapi", "tetapi", "sehingga", "waktu", "saat", "ketika"}
        PUNCT_SPLIT = {",", ".", "!", "?", ":", ";"}

        chunks: List[List[Dict[str, Any]]] = []
        current_chunk: List[Dict[str, Any]] = []

        n = len(words)
        for i, w_obj in enumerate(words):
            w_raw = str(w_obj.get("word", "")).strip()
            w_lower = re.sub(r"[^\w\s]", "", w_raw).lower()
            current_chunk.append(w_obj)

            ends_with_punct = bool(w_raw and w_raw[-1] in PUNCT_SPLIT)
            chunk_len = len(current_chunk)

            # Lookahead to next word
            next_lower = ""
            if i + 1 < n:
                next_raw = str(words[i + 1].get("word", "")).strip()
                next_lower = re.sub(r"[^\w\s]", "", next_raw).lower()

            should_split = False

            if ends_with_punct:
                should_split = True
            elif chunk_len >= max_words:
                # If current word is a negation or preposition, try not to split right here if possible
                if (w_lower in NEGATIONS or w_lower in PREPOSITIONS) and chunk_len < max_words + 1 and i + 1 < n:
                    should_split = False
                else:
                    should_split = True
            elif chunk_len >= 2:
                # If next word is a conjunction, it naturally opens the next chunk
                if next_lower in CONJUNCTIONS:
                    should_split = True
                # If next word is a negation or preposition, split before it so the new chunk starts with negation/prep
                elif next_lower in NEGATIONS or next_lower in PREPOSITIONS:
                    should_split = True
                # If current word is NOT negation or preposition, and chunk reached 3 words
                elif chunk_len >= 3 and next_lower not in NEGATIONS and next_lower not in PREPOSITIONS:
                    should_split = True

            if should_split or i == n - 1:
                chunks.append(current_chunk)
                current_chunk = []

        if current_chunk:
            if chunks and len(current_chunk) == 1 and len(chunks[-1]) < max_words + 1:
                chunks[-1].extend(current_chunk)
            else:
                chunks.append(current_chunk)

        return chunks

    def generate_ass(
        self,
        subtitle_data: List[Dict[str, Any]],
        edit_plan: EditPlan,
        output_path: Path,
        has_word_timestamps: bool = True
    ) -> Path:
        """
        Creates ASS file based on EditPlan subtitle style.
        If has_word_timestamps is True, creates clean natural word-highlighted chunks.
        If False, creates calm static phrase-level dialogue lines per segment.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        style: SubtitleStyle = edit_plan.subtitle_style
        emphasis_words = set(w.lower() for w in edit_plan.emphasis_words)

        # Header template
        font_name = style.font_name or "Montserrat-Black"
        ass_header = f"""[Script Info]
Title: Auto Short Subtitles V2
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{font_name},{style.font_size},{style.primary_color},&H000000FF,{style.outline_color},&H90000000,-1,0,0,0,100,100,1,0,1,{style.outline_width},{style.shadow_width},2,100,120,{style.margin_v},1
Style: Highlight,{font_name},{style.font_size},{style.highlight_color},&H000000FF,{style.outline_color},&H90000000,-1,0,0,0,100,100,1,0,1,{style.outline_width},{style.shadow_width},2,100,120,{style.margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
        dialogue_lines: List[str] = []

        if has_word_timestamps:
            # Flatten words from segments
            all_words = []
            for seg in subtitle_data:
                words = seg.get("words", [])
                if words:
                    for w in words:
                        all_words.append(w)
                else:
                    # Fallback to single chunk
                    all_words.append({
                        "word": seg.get("text", ""),
                        "start": seg.get("start", 0.0),
                        "end": seg.get("end", seg.get("start", 0.0) + 1.0)
                    })

            # Chunk into natural Indonesian phrase groups (respecting negations, prepositions, conjunctions)
            chunk_size = max(2, min(5, style.max_words_per_line))
            word_chunks = self.chunk_indonesian_words(all_words, max_words=chunk_size)

            for chunk in word_chunks:
                if not chunk:
                    continue
                c_start = chunk[0].get("start", 0.0)
                c_end = chunk[-1].get("end", c_start + 1.0)

                # Clamp to clip duration
                if c_start >= edit_plan.clip_duration:
                    continue
                c_end = min(edit_plan.clip_duration, c_end)
                if c_end <= c_start:
                    continue

                # Build line with clean, calm formatting
                # Only words in emphasis_words get highlighted color
                parts = []
                for w_obj in chunk:
                    w_text = escape_ass_text(str(w_obj.get("word", "")).strip())
                    clean_lookup = re.sub(r"[^\w\s]", "", w_text).lower()
                    if clean_lookup in emphasis_words:
                        # Highlighted color
                        parts.append(f"{{\\c{style.highlight_color}}}{w_text}{{\\c{style.primary_color}}}")
                    else:
                        parts.append(w_text)

                line_text = " ".join(parts).strip()
                t_start_str = format_ass_time(c_start)
                t_end_str = format_ass_time(c_end)
                dialogue_lines.append(f"Dialogue: 0,{t_start_str},{t_end_str},Default,,0,0,0,,{line_text}")

        else:
            # Segment-only mode (e.g. YouTube API fallback)
            # Show phrase-level static subtitles without synthesizing fake word animation.
            # Enforce deterministic non-overlapping timings (end = min(end, next_start))
            # so that no two dialogue blocks appear simultaneously on screen.
            valid_segs = []
            for seg in subtitle_data:
                s_start = float(seg.get("start", 0.0))
                s_end = float(seg.get("end", s_start + seg.get("duration", 2.0)))
                raw_text = escape_ass_text(str(seg.get("text", "")).strip())
                if s_start >= edit_plan.clip_duration or not raw_text:
                    continue
                s_end = min(edit_plan.clip_duration, s_end)
                if s_end <= s_start:
                    continue
                valid_segs.append({
                    "start": s_start,
                    "end": s_end,
                    "text": raw_text
                })

            # Sort by start time
            valid_segs.sort(key=lambda s: s["start"])

            for idx, cur_seg in enumerate(valid_segs):
                s_start = cur_seg["start"]
                s_end = cur_seg["end"]
                # If next segment starts before current segment ends, clamp current end to next start
                if idx + 1 < len(valid_segs):
                    next_start = valid_segs[idx + 1]["start"]
                    if next_start > s_start:
                        s_end = min(s_end, next_start)
                    else:
                        # Identical start time edge-case: keep minimal non-zero duration
                        s_end = max(s_start + 0.1, min(s_end, next_start))

                if s_end <= s_start:
                    continue

                t_start_str = format_ass_time(s_start)
                t_end_str = format_ass_time(s_end)
                dialogue_lines.append(f"Dialogue: 0,{t_start_str},{t_end_str},Default,,0,0,0,,{cur_seg['text']}")

        content = ass_header + "\n".join(dialogue_lines) + "\n"
        output_path.write_text(content, encoding="utf-8")
        return output_path


subtitle_generator = SubtitleGeneratorV2()
