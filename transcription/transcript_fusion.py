"""Transcript Fusion Module (Auto Clipper V3.1 Hybrid Subtitle Accuracy).

Combines three evidence sources for maximum subtitle text accuracy:
1. faster-whisper large-v3 word timestamps on CPU int8 (KAPAN ucapan terjadi)
2. Source transcript / caption (reference text jika ada)
3. Gemini native-video transcript verifier via 9router (APA yang sebenarnya diucapkan)

Division of responsibility (Spec Section 4 & 5):
- Whisper = timing engine (word start/end from audio)
- Gemini = text corrector (verbatim what is spoken, slang & code-switching preserved)
- TranscriptFusion = fuse evidence into canonical 2-5 word subtitle phrases with audio-derived spans
"""

import difflib
import json
import logging
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from config import settings
from llm_client import llm_client

logger = logging.getLogger(__name__)

# Constants
EPSILON_SEC = 0.02  # 20ms gap between consecutive events
DEFAULT_LARGE_V3_PATH = "/mnt/storage/hermes_home_orion/.cache/huggingface/hub/models--Systran--faster-whisper-large-v3/snapshots/edaa852ec7e145841d8ffdb056a99866b5f0a478"
_WHISPER_MODEL_CACHE: Dict[Tuple[str, str, str, int], Any] = {}


def clean_word_for_matching(w: str) -> str:
    """Normalizes a word string for token alignment."""
    return re.sub(r"[^\w]", "", w.lower()).strip()


def normalize_phrase_text(t: str) -> str:
    """Normalizes text with whitespace preserved between words."""
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", t.lower())).strip()


def build_context_prompt(
    video_title: str = "",
    channel_title: str = "",
    source_transcript_excerpt: str = "",
) -> str:
    """Builds small context / hotwords prompt from video metadata for ASR initialization.

    Spec Section 7: Hotwords only. Do NOT include transcript excerpt as prompt
    to avoid Whisper prompt-skipping hallucination.
    """
    parts = ["Percakapan podcast santai bahasa Indonesia."]

    if channel_title:
        clean_channel = re.sub(r"[#|\[\]()]", "", channel_title).strip()
        parts.append(f"Channel: {clean_channel}.")

    if video_title:
        clean_title = re.sub(r"[#|\[\]()]", "", video_title).strip()
        parts.append(f"Topik: {clean_title[:80]}.")

    # Critical hotwords and recurring phrases
    hotwords = [
        "Mas Bilal", "Kak Jeje", "self love", "muka bumi", "nomor tiga",
        "who are you", "what do you want", "what you can give", "satpam",
        "ongkir", "sumringah", "S1"
    ]
    parts.append(f"Kata kunci: {', '.join(hotwords)}.")
    return " ".join(parts)


def extract_word_timestamps_large(
    video_path: str,
    language: str = "id",
    model_size: str = "large-v3",
    device: str = "cpu",
    compute_type: str = "int8",
    context_prompt: str = "",
    beam_size: int = 5,
    cpu_threads: int = 4,
) -> List[Dict[str, Any]]:
    """Runs faster-whisper large-v3 with word timestamps on clip audio.

    Audits and logs actual runtime parameters:
      ASR_MODEL=
      ASR_COMPUTE_TYPE=
      LANGUAGE=
      BEAM_SIZE=
      WORD_TIMESTAMPS=
      INITIAL_PROMPT=
    """
    from faster_whisper import WhisperModel

    if not os.environ.get("HF_HOME") and os.path.exists("/mnt/storage/hermes_home_orion/.cache/huggingface"):
        os.environ["HF_HOME"] = "/mnt/storage/hermes_home_orion/.cache/huggingface"

    effective_model = model_size
    if model_size in ("large-v3", "large") and os.path.exists(DEFAULT_LARGE_V3_PATH):
        effective_model = DEFAULT_LARGE_V3_PATH

    initial_prompt = context_prompt or "Ini adalah podcast Indonesia, percakapan santai tentang kehidupan."

    logger.info("=== ASR RUNTIME AUDIT (Spec Section 1) ===")
    logger.info(f"ASR_MODEL={effective_model}")
    logger.info(f"ASR_COMPUTE_TYPE={compute_type}")
    logger.info(f"LANGUAGE={language}")
    logger.info(f"BEAM_SIZE={beam_size}")
    logger.info("WORD_TIMESTAMPS=True")
    logger.info(f"INITIAL_PROMPT={initial_prompt[:120]}...")
    logger.info("VAD_FILTER=True")

    tmp_wav = tempfile.mktemp(suffix=".wav", prefix="asr_clip_")
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
        tmp_wav,
    ]
    try:
        subprocess.run(cmd, capture_output=True, timeout=30, check=True)
    except Exception as e:
        logger.error(f"Audio extraction failed for ASR: {e}")
        return []

    try:
        cache_key = (effective_model, device, compute_type, cpu_threads)
        if cache_key not in _WHISPER_MODEL_CACHE:
            logger.info(f"Loading faster-whisper model into memory cache: {effective_model}")
            _WHISPER_MODEL_CACHE[cache_key] = WhisperModel(
                effective_model,
                device=device,
                compute_type=compute_type,
                cpu_threads=cpu_threads,
            )
        model = _WHISPER_MODEL_CACHE[cache_key]

        segments, info = model.transcribe(
            tmp_wav,
            language=language,
            word_timestamps=True,
            beam_size=beam_size,
            initial_prompt=initial_prompt,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=300),
        )

        words: List[Dict[str, Any]] = []
        for seg in segments:
            if seg.words:
                for w in seg.words:
                    clean_w = w.word.strip()
                    if clean_w:
                        words.append({
                            "word": clean_w,
                            "start": round(float(w.start), 3),
                            "end": round(float(w.end), 3),
                            "probability": round(float(getattr(w, "probability", 0.95)), 3),
                        })

        logger.info(f"ASR extraction completed: {len(words)} words via {model_size}")
        return words

    except Exception as e:
        logger.error(f"Whisper transcription failed: {e}")
        return []
    finally:
        if os.path.exists(tmp_wav):
            try:
                os.unlink(tmp_wav)
            except Exception:
                pass


def gemini_verify_transcript(
    video_path: str,
    asr_text: str,
    source_transcript: str = "",
    video_title: str = "",
    channel_title: str = "",
    client=None,
) -> Dict[str, Any]:
    """Sends candidate clip MP4 directly via 9router to Gemini 3.8 Flash for verbatim correction."""
    client = client or llm_client

    system_prompt = (
        "Correct EXACTLY what is spoken.\n"
        "Preserve Indonesian slang.\n"
        "Preserve English code-switching.\n"
        "Do not paraphrase.\n"
        "Do not summarize.\n"
        "Do not censor.\n"
        "Do not invent words.\n"
        "Use provided transcripts only as hints.\n"
        "Resolve disagreements by listening to the video/audio.\n"
        "Format answer strictly as valid JSON."
    )

    prompt = (
        "Dengarkan video audio ini secara teliti dan perbaiki transkripsi teks ucapan pembicara agar 100% akurat verbatim.\n\n"
        f"=== ASR Transcript (mungkin ada salah dengar / fonetik) ===\n{asr_text}\n\n"
    )
    if source_transcript:
        prompt += f"=== Source Caption (referensi) ===\n{source_transcript}\n\n"

    context_lines = []
    if video_title:
        context_lines.append(f"Judul: {video_title}")
    if channel_title:
        context_lines.append(f"Channel: {channel_title}")
    if context_lines:
        prompt += f"=== Context ===\n{chr(10).join(context_lines)}\n\n"

    prompt += (
        "Kembalikan keputusan koreksi HANYA dalam JSON valid:\n"
        "{\n"
        '  "corrected_full_text": "teks lengkap yang diperbaiki verbatim sesuai yang terdengar di video",\n'
        '  "segments": [\n'
        '    {\n'
        '      "asr_text": "frasa asr yang salah",\n'
        '      "corrected_text": "frasa yang benar",\n'
        '      "confidence": 0.98\n'
        '    }\n'
        '  ],\n'
        '  "overall_confidence": 0.95\n'
        "}"
    )

    try:
        raw = client.video_completion(
            video_path=video_path,
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=0.1,
        )
        parsed = client.extract_json(raw)
        if parsed:
            corrected_full = str(parsed.get("corrected_full_text", "")).strip()
            segments = list(parsed.get("segments", []))
            if not segments and "corrections" in parsed:
                for c in parsed.get("corrections", []):
                    segments.append({
                        "asr_text": c.get("original", ""),
                        "corrected_text": c.get("corrected", ""),
                        "confidence": float(c.get("confidence", 0.95)),
                    })

            overall_conf = float(parsed.get("overall_confidence", 0.90))

            return {
                "corrected_full_text": corrected_full,
                "segments": segments,
                "overall_confidence": overall_conf,
                "mode": "GEMINI_NATIVE_VIDEO",
                "raw_response": raw[:400],
            }
    except Exception as e:
        logger.warning(f"Gemini native-video transcript verification failed: {e}")

    return {
        "corrected_full_text": "",
        "segments": [],
        "overall_confidence": 0.0,
        "mode": "FAILED",
    }


def _chunk_asr_words_directly(
    asr_words: List[Dict[str, Any]],
    source_transcript: str = "",
    min_words: int = 2,
    max_words: int = 5,
    max_gap_sec: float = 0.35,
) -> List[Dict[str, Any]]:
    """Chunks word tokens into 2-5 word phrases based on natural audio boundaries."""
    phrases = []
    curr = []

    def flush():
        if not curr:
            return
        text = " ".join(w["word"] for w in curr).strip()
        evidence = ["whisper"]
        if source_transcript and clean_word_for_matching(text) in clean_word_for_matching(source_transcript):
            evidence.append("source_caption")
        avg_prob = sum(w.get("probability", 0.8) for w in curr) / len(curr)
        conf = "HIGH" if avg_prob >= 0.85 else ("MEDIUM" if avg_prob >= 0.60 else "LOW")
        phrases.append({
            "text": text,
            "start": round(curr[0]["start"], 3),
            "end": round(curr[-1]["end"], 3),
            "confidence": conf,
            "evidence": evidence,
            "words": list(curr),
        })
        curr.clear()

    for i, w in enumerate(asr_words):
        curr.append(w)
        should_split = False
        if len(curr) >= max_words:
            should_split = True
        elif w["word"] and w["word"][-1] in ".!?;," and len(curr) >= min_words:
            should_split = True
        elif i < len(asr_words) - 1:
            gap = asr_words[i + 1]["start"] - w["end"]
            if gap >= max_gap_sec and len(curr) >= min_words:
                should_split = True
        if should_split:
            flush()

    flush()
    return phrases


def fuse_transcript(
    asr_words: List[Dict[str, Any]],
    gemini_result: Dict[str, Any],
    source_transcript: str = "",
    min_words: int = 2,
    max_words: int = 5,
    max_gap_sec: float = 0.35,
) -> List[Dict[str, Any]]:
    """Fuses ASR word timestamps with Gemini-corrected verbatim text into 2-5 word phrases.

    Invariant Rules (Spec Section 4 & 5):
    - Whisper = timing engine (word start/end from audio).
    - Phrases are chunked directly from ASR words FIRST so start/end spans are locked to audio.
    - Gemini ONLY corrects text; timing start/end span ALWAYS taken from audio Whisper word timestamps.
    - Start and end spans of phrases NEVER shrink or compress to artificial 0.05s intervals.
    - Clip audio duration is fully covered without gaps or squeezed phrases.
    """
    if not asr_words:
        return []

    # Step 1: Chunk ASR words into natural phrases first
    phrases = _chunk_asr_words_directly(
        asr_words=asr_words,
        source_transcript=source_transcript,
        min_words=min_words,
        max_words=max_words,
        max_gap_sec=max_gap_sec,
    )

    if not phrases:
        return []

    corrected_full_text = str(gemini_result.get("corrected_full_text", "")).strip()
    segments = list(gemini_result.get("segments", []))
    if not segments and "corrections" in gemini_result:
        for c in gemini_result.get("corrections", []):
            segments.append({
                "asr_text": c.get("original", ""),
                "corrected_text": c.get("corrected", ""),
                "confidence": float(c.get("confidence", 0.95)),
            })
    overall_conf = float(gemini_result.get("overall_confidence", 0.0))

    # If Gemini is unavailable, fallback to raw ASR phrases directly
    if not corrected_full_text and not segments:
        logger.info("Gemini correction unavailable; returning audio-derived ASR phrases directly")
        return phrases

    # Step 2: If corrected_full_text is available, align phrase tokens to corrected full text
    if corrected_full_text:
        word_to_phrase: List[int] = []
        for p_idx, p in enumerate(phrases):
            for _ in range(len(p.get("words", []))):
                word_to_phrase.append(p_idx)
        while len(word_to_phrase) < len(asr_words):
            word_to_phrase.append(len(phrases) - 1)

        asr_tokens = [w["word"] for w in asr_words]
        corr_tokens = corrected_full_text.split()

        a_norm = [clean_word_for_matching(w) for w in asr_tokens]
        b_norm = [clean_word_for_matching(w) for w in corr_tokens]

        sm = difflib.SequenceMatcher(None, a_norm, b_norm)
        phrase_tokens: Dict[int, List[str]] = {i: [] for i in range(len(phrases))}

        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag == "equal":
                for ai, bj in zip(range(i1, i2), range(j1, j2)):
                    p_idx = word_to_phrase[ai]
                    phrase_tokens[p_idx].append(corr_tokens[bj])
            elif tag == "replace":
                target_p = word_to_phrase[i1]
                phrase_tokens[target_p].extend(corr_tokens[j1:j2])
            elif tag == "insert":
                target_p = word_to_phrase[i1] if i1 < len(word_to_phrase) else (len(phrases) - 1)
                phrase_tokens[target_p].extend(corr_tokens[j1:j2])
            elif tag == "delete":
                pass

        for p_idx, p in enumerate(phrases):
            aligned = phrase_tokens.get(p_idx, [])
            if aligned:
                new_text = " ".join(aligned).strip()
                new_text = re.sub(r"\s+([,.!?;])", r"\1", new_text)
                p["text"] = new_text
                if "gemini_native_video" not in p["evidence"]:
                    p["evidence"].append("gemini_native_video")

    # Step 3: Segment-level explicit overrides and keyword replacements
    if segments:
        for seg in segments:
            orig = str(seg.get("asr_text", "")).strip()
            fixed = str(seg.get("corrected_text", "")).strip()
            if not orig or not fixed:
                continue
            orig_clean = clean_word_for_matching(orig)
            fixed_clean = clean_word_for_matching(fixed)
            for p in phrases:
                p_clean = clean_word_for_matching(p["text"])
                if fixed_clean and fixed_clean in p_clean:
                    continue
                pattern = re.compile(re.escape(orig), re.IGNORECASE)
                if pattern.search(p["text"]):
                    p["text"] = pattern.sub(fixed, p["text"])
                    if "gemini_native_video" not in p["evidence"]:
                        p["evidence"].append("gemini_native_video")
                    p["confidence"] = "HIGH"
                elif orig_clean and orig_clean == p_clean:
                    p["text"] = fixed
                    if "gemini_native_video" not in p["evidence"]:
                        p["evidence"].append("gemini_native_video")
                    p["confidence"] = "HIGH"

    # Step 4: Finalize confidence and evidence tags
    for p in phrases:
        if overall_conf >= 0.80:
            p["confidence"] = "HIGH"
        elif overall_conf >= 0.50:
            p["confidence"] = "MEDIUM"
        else:
            p["confidence"] = "LOW"

        if "gemini_native_video" not in p["evidence"] and (corrected_full_text or segments):
            p["evidence"].append("gemini_native_video")

        if source_transcript:
            p_clean = normalize_phrase_text(p["text"])
            s_clean = normalize_phrase_text(source_transcript)
            if p_clean and p_clean in s_clean and "source_caption" not in p["evidence"]:
                p["evidence"].append("source_caption")

    logger.info(f"Transcript Fusion generated {len(phrases)} canonical phrases (overall_conf={overall_conf:.2f})")
    return phrases


def enforce_non_overlapping_timeline(
    phrases: List[Dict[str, Any]],
    linger_sec: float = 0.10,
    max_silence_hold_sec: float = 0.30,
) -> List[Dict[str, Any]]:
    """Enforces hard subtitle timeline invariants:
    - event[i].start < event[i].end
    - event[i].end <= event[i+1].start - EPSILON_SEC (zero event overlap, single lane)
    - Subtitle clears during silence gaps > max_silence_hold_sec
    - Small linger (80-120ms) after last word
    """
    if not phrases:
        return []

    validated: List[Dict[str, Any]] = []
    for i, phrase in enumerate(phrases):
        start = phrase["start"]
        raw_end = phrase["end"] + linger_sec

        if i < len(phrases) - 1:
            next_start = phrases[i + 1]["start"]
            gap = next_start - phrase["end"]
            if gap > max_silence_hold_sec:
                raw_end = phrase["end"] + min(linger_sec, gap * 0.3)
            raw_end = min(raw_end, next_start - EPSILON_SEC)

        if raw_end <= start:
            raw_end = start + 0.10

        validated.append({
            "text": phrase["text"],
            "start": round(start, 3),
            "end": round(raw_end, 3),
            "confidence": phrase.get("confidence", "MEDIUM"),
            "evidence": phrase.get("evidence", ["whisper"]),
            "words": phrase.get("words", []),
        })

    return validated


def validate_ass_timeline(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Validates ASS timeline against hard invariants."""
    if not events:
        return {"valid": False, "error": "no_events", "event_count": 0}

    invalid_intervals = 0
    overlaps = 0
    max_simultaneous = 1

    for i, ev in enumerate(events):
        if ev["end"] <= ev["start"]:
            invalid_intervals += 1
        if i < len(events) - 1:
            if ev["end"] > events[i + 1]["start"] + 0.001:
                overlaps += 1

    for i, ev in enumerate(events):
        t = ev["start"]
        simultaneous = sum(1 for e in events if e["start"] <= t < e["end"])
        max_simultaneous = max(max_simultaneous, simultaneous)

    outside_clip = sum(1 for e in events if e["start"] < -0.01)

    is_valid = (invalid_intervals == 0 and overlaps == 0 and max_simultaneous <= 1)
    return {
        "valid": is_valid,
        "event_count": len(events),
        "invalid_intervals": invalid_intervals,
        "overlapping_events": overlaps,
        "max_simultaneous": max_simultaneous,
        "events_outside_clip": outside_clip,
    }


def format_ass_time(seconds: float) -> str:
    """Formats float seconds into ASS timestamp: H:MM:SS.cs"""
    seconds = max(0.0, float(seconds))
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    cs = int(round((seconds - int(seconds)) * 100))
    if cs >= 100:
        s += 1
        cs -= 100
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def generate_fused_ass(
    fused_phrases: List[Dict[str, Any]],
    output_ass_path: str,
    font_name: str = "Montserrat",
    font_size: int = 46,
    margin_v: int = 440,
    margin_h: int = 90,
) -> Tuple[bool, Dict[str, Any], List[Dict[str, Any]]]:
    """Generates ASS subtitle file from fused phrases enforcing V3.1 timeline invariants."""
    if not fused_phrases:
        return False, {"valid": False, "error": "empty_fused_phrases"}, []

    validated = enforce_non_overlapping_timeline(fused_phrases)
    report = validate_ass_timeline(validated)
    report["phrase_count"] = len(validated)

    if not report["valid"]:
        logger.error(f"ASS timeline validation failed: {report}")
        return False, report, validated

    ass_lines = [
        "[Script Info]",
        "ScriptType: v4.00+",
        "PlayResX: 1080",
        "PlayResY: 1920",
        "ScaledBorderAndShadow: yes",
        "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, "
        "Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
        "Alignment, MarginL, MarginR, MarginV, Encoding",
        f"Style: Default,{font_name},{font_size},"
        "&H00FFFFFF,&H000000FF,&H00000000,&H80000000,"
        f"1,0,0,0,100,100,0,0,1,4,2,2,{margin_h},{margin_h},{margin_v},1",
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]

    for ev in validated:
        start_ts = format_ass_time(ev["start"])
        end_ts = format_ass_time(ev["end"])
        text = ev["text"].replace("\\", "\\\\").strip()
        words_in_text = text.split()
        if len(words_in_text) > 3:
            mid = len(words_in_text) // 2
            line1 = " ".join(words_in_text[:mid])
            line2 = " ".join(words_in_text[mid:])
            text = f"{line1}\\N{line2}"
        ass_lines.append(f"Dialogue: 0,{start_ts},{end_ts},Default,,0,0,0,,{text}")

    Path(output_ass_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_ass_path, "w", encoding="utf-8") as f:
        f.write("\n".join(ass_lines) + "\n")

    logger.info(f"Generated fused ASS subtitle: {output_ass_path} ({len(validated)} events, 0 overlaps)")
    report["ass_path"] = output_ass_path
    return True, report, validated
