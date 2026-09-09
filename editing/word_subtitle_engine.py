"""Word-Level Subtitle Generator (V3.1 Subtitle Engine Stabilization).

Generates clean ASS subtitles from clip-local faster-whisper word timestamps.
Hard invariant: ZERO event overlap, ONE caption lane, word-derived timing only.

Pipeline:
1. Extract audio from candidate clip
2. Run faster-whisper with word_timestamps=True, language='id'
3. Chunk words into 2-5 word phrases at natural boundaries
4. Enforce non-overlapping timeline with gap clearing
5. Generate ASS with validated timeline
"""

import json
import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

EPSILON_SEC = 0.02  # 20ms gap between events


def extract_word_timestamps(
    video_path: str,
    language: str = "id",
    model_size: str = "base",
    device: str = "cpu",
    compute_type: str = "int8",
) -> List[Dict[str, Any]]:
    """Runs faster-whisper on the clip audio and returns word-level timestamps.

    Returns list of: [{"word": "gue", "start": 1.20, "end": 1.36}, ...]
    """
    from faster_whisper import WhisperModel

    # Extract audio to temp WAV
    tmp_wav = tempfile.mktemp(suffix=".wav")
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
        tmp_wav,
    ]
    try:
        subprocess.run(cmd, capture_output=True, timeout=30, check=True)
    except Exception as e:
        logger.error(f"Audio extraction failed: {e}")
        return []

    try:
        model = WhisperModel(model_size, device=device, compute_type=compute_type)
        segments, info = model.transcribe(
            tmp_wav,
            language=language,
            word_timestamps=True,
            beam_size=5,
            initial_prompt="Ini adalah podcast Indonesia, percakapan santai tentang kehidupan.",
            vad_filter=True,
        )

        words = []
        for seg in segments:
            if seg.words:
                for w in seg.words:
                    words.append({
                        "word": w.word.strip(),
                        "start": round(w.start, 3),
                        "end": round(w.end, 3),
                    })
        logger.info(f"Extracted {len(words)} word tokens from clip via faster-whisper ({model_size})")
        return words

    except Exception as e:
        logger.error(f"Whisper transcription failed: {e}")
        return []
    finally:
        if os.path.exists(tmp_wav):
            os.unlink(tmp_wav)


def chunk_words_to_phrases(
    words: List[Dict[str, Any]],
    min_words: int = 2,
    max_words: int = 5,
    max_gap_sec: float = 0.35,
) -> List[Dict[str, Any]]:
    """Groups word tokens into 2-5 word caption phrases.

    Splits at:
    - Punctuation (., !, ?, ;)
    - Natural pause (gap > max_gap_sec between words)
    - Max word count reached

    Returns: [{"text": "gue waktu itu", "start": 1.20, "end": 1.81, "words": [...]}, ...]
    """
    if not words:
        return []

    phrases = []
    current_words = []

    def flush_phrase():
        if not current_words:
            return
        text = " ".join(w["word"] for w in current_words)
        phrases.append({
            "text": text,
            "start": current_words[0]["start"],
            "end": current_words[-1]["end"],
            "words": list(current_words),
        })
        current_words.clear()

    for i, w in enumerate(words):
        word_text = w["word"]
        current_words.append(w)

        # Check split conditions
        should_split = False

        # 1. Max words reached
        if len(current_words) >= max_words:
            should_split = True

        # 2. Punctuation at end of word
        elif word_text and word_text[-1] in ".!?;":
            if len(current_words) >= min_words:
                should_split = True

        # 3. Natural pause before next word
        elif i < len(words) - 1:
            gap = words[i + 1]["start"] - w["end"]
            if gap >= max_gap_sec and len(current_words) >= min_words:
                should_split = True

        if should_split:
            flush_phrase()

    # Flush remaining
    flush_phrase()

    return phrases


def enforce_non_overlapping_timeline(
    phrases: List[Dict[str, Any]],
    linger_sec: float = 0.10,
    max_silence_hold_sec: float = 0.30,
) -> List[Dict[str, Any]]:
    """Enforces the hard subtitle timeline invariant:

    - event[i].start < event[i].end
    - event[i].end <= event[i+1].start
    - Subtitle clears during silence gaps > max_silence_hold_sec
    - Small linger (80-150ms) after last word
    """
    if not phrases:
        return []

    validated = []
    for i, phrase in enumerate(phrases):
        start = phrase["start"]
        raw_end = phrase["end"] + linger_sec

        if i < len(phrases) - 1:
            next_start = phrases[i + 1]["start"]
            # Check for silence gap
            gap = next_start - phrase["end"]
            if gap > max_silence_hold_sec:
                # Clear subtitle during silence
                raw_end = phrase["end"] + min(linger_sec, gap * 0.3)
            # Clamp: end <= next_start - epsilon
            raw_end = min(raw_end, next_start - EPSILON_SEC)

        # Ensure valid interval
        if raw_end <= start:
            raw_end = start + 0.1

        validated.append({
            "text": phrase["text"],
            "start": round(start, 3),
            "end": round(raw_end, 3),
            "words": phrase.get("words", []),
        })

    return validated


def validate_ass_timeline(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Validates the ASS timeline against hard invariants.

    Returns validation report with overlap count, max simultaneous events, etc.
    """
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

    # Check max simultaneous at each event boundary
    for i, ev in enumerate(events):
        t = ev["start"]
        simultaneous = sum(1 for e in events if e["start"] <= t < e["end"])
        max_simultaneous = max(max_simultaneous, simultaneous)

    outside_clip = sum(1 for e in events if e["start"] < -0.01)

    return {
        "valid": invalid_intervals == 0 and overlaps == 0 and max_simultaneous <= 1,
        "event_count": len(events),
        "invalid_intervals": invalid_intervals,
        "overlapping_events": overlaps,
        "max_simultaneous": max_simultaneous,
        "events_outside_clip": outside_clip,
    }


def _prepare_candidate_clip(
    video_path: str,
    start_sec: float = 0.0,
    duration_sec: Optional[float] = None,
) -> Tuple[str, bool]:
    """Ensures candidate clip slice is ready if start_sec > 0 or duration_sec is set.
    Uses fast stream-copy slicing to ensure speed (<0.5s) and exact temporal boundary.
    Returns (effective_video_path, is_temporary).
    """
    if start_sec <= 0.001 and duration_sec is None:
        return video_path, False

    tmp_slice = tempfile.mktemp(suffix=".mp4", prefix="hybrid_sub_slice_")
    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{start_sec:.3f}",
        "-i", video_path,
    ]
    if duration_sec is not None and duration_sec > 0:
        cmd.extend(["-t", f"{duration_sec:.3f}"])
    cmd.extend([
        "-c", "copy",
        "-movflags", "+faststart",
        tmp_slice,
    ])
    try:
        subprocess.run(cmd, capture_output=True, timeout=30, check=True)
        if os.path.exists(tmp_slice) and os.path.getsize(tmp_slice) > 1000:
            return tmp_slice, True
        raise RuntimeError(f"Sliced candidate clip is empty or invalid: {tmp_slice}")
    except Exception as e:
        logger.error(f"Candidate slicing failed ({e}) for {video_path} [start={start_sec}, dur={duration_sec}]")
        if start_sec > 0.001:
            raise RuntimeError(
                f"Candidate slicing failed for {video_path} at start_sec={start_sec}. "
                f"Cannot fallback to full video because transcription would be completely wrong: {e}"
            ) from e
        return video_path, False


def generate_hybrid_subtitles(
    video_path: str,
    output_ass_path: str,
    start_sec: float = 0.0,
    duration_sec: Optional[float] = None,
    source_transcript_excerpt: str = "",
    video_title: str = "",
    channel_title: str = "",
    model_size: str = "large-v3",
    debug_timeline_path: Optional[str] = None,
    font_name: str = "Montserrat",
    font_size: int = 46,
    margin_v: int = 440,
    margin_h: int = 90,
) -> Tuple[bool, Dict[str, Any], List[Dict[str, Any]]]:
    """Generates ASS subtitles using Auto Clipper V3.1 Hybrid Subtitle Accuracy Engine.

    Steps:
    a. Extracts audio word timestamps using extract_word_timestamps_large from transcript_fusion
       (faster-whisper large-v3 on CPU int8, vad_filter=True, beam_size=5).
    b. Verifies transcript with Gemini via 9router using gemini_verify_transcript.
    c. Performs evidence fusion via fuse_transcript.
    d. Generates validated ASS via generate_fused_ass.
    e. Returns (success, report, fused_phrases).
    """
    from transcription.transcript_fusion import (
        build_context_prompt,
        extract_word_timestamps_large,
        gemini_verify_transcript,
        fuse_transcript,
        generate_fused_ass,
    )

    try:
        effective_path, is_temp = _prepare_candidate_clip(video_path, start_sec, duration_sec)
    except Exception as e:
        logger.error(f"Failed to prepare candidate clip: {e}")
        return False, {"valid": False, "error": f"slicing_failed: {e}"}, []

    try:
        context_prompt = build_context_prompt(
            video_title=video_title,
            channel_title=channel_title,
            source_transcript_excerpt=source_transcript_excerpt,
        )

        # a. Extract ASR word timestamps using faster-whisper large-v3 on CPU int8
        asr_words = extract_word_timestamps_large(
            video_path=effective_path,
            language="id",
            model_size=model_size,
            context_prompt=context_prompt,
            beam_size=5,
        )

        if not asr_words:
            logger.warning(f"No word tokens extracted from {effective_path}")
            return False, {"valid": False, "error": "no_words_extracted", "event_count": 0}, []

        asr_full_text = " ".join(w["word"] for w in asr_words)

        # b. Verify text with Gemini via 9router
        gemini_result = gemini_verify_transcript(
            video_path=effective_path,
            asr_text=asr_full_text,
            source_transcript=source_transcript_excerpt,
            video_title=video_title,
            channel_title=channel_title,
        )

        # c. Evidence fusion
        fused_phrases = fuse_transcript(
            asr_words=asr_words,
            gemini_result=gemini_result,
            source_transcript=source_transcript_excerpt,
        )

        if not fused_phrases:
            logger.warning(f"Transcript fusion produced 0 phrases for {effective_path}")
            return False, {"valid": False, "error": "fusion_failed", "event_count": 0}, []

        # d. Generate ASS
        ok, report, validated_phrases = generate_fused_ass(
            fused_phrases=fused_phrases,
            output_ass_path=output_ass_path,
            font_name=font_name,
            font_size=font_size,
            margin_v=margin_v,
            margin_h=margin_h,
        )

        report["word_count"] = len(asr_words)
        report["gemini_mode"] = gemini_result.get("mode", "UNKNOWN")
        report["gemini_confidence"] = gemini_result.get("overall_confidence", 0.0)

        if debug_timeline_path:
            Path(debug_timeline_path).parent.mkdir(parents=True, exist_ok=True)
            with open(debug_timeline_path, "w", encoding="utf-8") as f:
                json.dump(validated_phrases, f, indent=2, ensure_ascii=False)

        return ok, report, validated_phrases

    finally:
        if is_temp and os.path.exists(effective_path):
            try:
                os.unlink(effective_path)
            except Exception:
                pass


def generate_ass_from_words(
    video_path: str,
    output_ass_path: str,
    font_name: str = "Montserrat",
    font_size: int = 46,
    margin_v: int = 440,
    margin_h: int = 90,
    language: str = "id",
    model_size: str = "large-v3",
    debug_timeline_path: Optional[str] = None,
    start_sec: float = 0.0,
    duration_sec: Optional[float] = None,
    source_transcript: str = "",
    video_title: str = "",
    channel_title: str = "",
    use_hybrid: bool = True,
) -> Tuple[bool, Dict[str, Any]]:
    """Full pipeline: extract words -> chunk -> validate -> write ASS.
    When use_hybrid=True (default in V3.1), delegates to generate_hybrid_subtitles.
    Returns (success, report) where report contains timeline validation.
    """
    if use_hybrid:
        ok, report, _ = generate_hybrid_subtitles(
            video_path=video_path,
            output_ass_path=output_ass_path,
            start_sec=start_sec,
            duration_sec=duration_sec,
            source_transcript_excerpt=source_transcript,
            video_title=video_title,
            channel_title=channel_title,
            model_size=model_size,
            debug_timeline_path=debug_timeline_path,
            font_name=font_name,
            font_size=font_size,
            margin_v=margin_v,
            margin_h=margin_h,
        )
        return ok, report

    # Step 1: Extract word timestamps
    words = extract_word_timestamps(
        video_path, language=language, model_size="base" if model_size == "large-v3" else model_size
    )
    if not words:
        return False, {"error": "no_words_extracted", "word_count": 0}

    # Step 2: Chunk into phrases
    phrases = chunk_words_to_phrases(words)
    if not phrases:
        return False, {"error": "no_phrases_generated", "word_count": len(words)}

    # Step 3: Enforce non-overlapping timeline
    validated = enforce_non_overlapping_timeline(phrases)

    # Step 4: Validate
    report = validate_ass_timeline(validated)
    report["word_count"] = len(words)
    report["phrase_count"] = len(validated)

    if not report["valid"]:
        logger.error(f"ASS timeline validation FAILED: {report}")
        return False, report

    # Step 5: Write ASS
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
        f"&H00FFFFFF,&H000000FF,&H00000000,&H80000000,"
        f"1,0,0,0,100,100,0,0,1,4,2,2,{margin_h},{margin_h},{margin_v},1",
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]

    for ev in validated:
        start_ts = _format_ass_time(ev["start"])
        end_ts = _format_ass_time(ev["end"])
        text = ev["text"].replace("\\", "\\\\").strip()
        # Split long phrases into 2 lines
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

    logger.info(f"Generated ASS subtitle: {output_ass_path} ({len(validated)} events, 0 overlaps)")

    # Optional debug timeline
    if debug_timeline_path:
        Path(debug_timeline_path).parent.mkdir(parents=True, exist_ok=True)
        with open(debug_timeline_path, "w") as f:
            json.dump(validated, f, indent=2, ensure_ascii=False)

    report["ass_path"] = output_ass_path
    return True, report


def _format_ass_time(seconds: float) -> str:
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
